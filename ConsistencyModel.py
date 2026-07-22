import torch
from torch import nn
import math
import copy
from Model import Denoise


class ConsistencyDenoise(nn.Module):
	"""Consistency Model denoiser bọc mạng Denoise gốc.

	Cấu trúc skip-connection đảm bảo điều kiện biên f(x, sigma_min) = x:
		f_theta(x_t, t) = c_skip(t) * x_t + c_out(t) * F_theta(x_t, t)

	Args:
		in_dims, out_dims, emb_size: tham số kiến trúc MLP (giống Denoise gốc)
		sigma_data: độ lệch chuẩn thực tế của dữ liệu (tính từ tập huấn luyện)
		sigma_min: mức nhiễu nhỏ nhất (epsilon trong công thức)
	"""

	def __init__(self, in_dims, out_dims, emb_size, sigma_data,
				 sigma_min=0.002, norm=False, dropout=0.5):
		super(ConsistencyDenoise, self).__init__()
		self.sigma_data = sigma_data
		self.sigma_min = sigma_min
		# Tái sử dụng kiến trúc Denoise gốc (MLP + sinusoidal time embedding)
		self.net = Denoise(in_dims, out_dims, emb_size, norm, dropout)

	def _c_skip(self, sigma):
		"""c_skip(t) = sigma_data^2 / ((t - eps)^2 + sigma_data^2)
		Tại t=sigma_min: c_skip = 1 (trả về input nguyên vẹn)
		"""
		return self.sigma_data ** 2 / (
			(sigma - self.sigma_min) ** 2 + self.sigma_data ** 2
		)

	def _c_out(self, sigma):
		"""c_out(t) = sigma_data * (t - eps) / sqrt(t^2 + sigma_data^2)
		Tại t=sigma_min: c_out = 0 (triệt tiêu output mạng)
		"""
		return (
			self.sigma_data * (sigma - self.sigma_min)
			/ torch.sqrt(sigma ** 2 + self.sigma_data ** 2)
		)

	def forward(self, x, sigma, mess_dropout=True):
		"""
		Args:
			x: input có nhiễu [batch_size, num_items]
			sigma: mức nhiễu liên tục [batch_size]
			mess_dropout: bật/tắt dropout
		Returns:
			output khử nhiễu [batch_size, num_items]
		"""
		c_skip = self._c_skip(sigma).unsqueeze(-1)  # [batch, 1]
		c_out = self._c_out(sigma).unsqueeze(-1)     # [batch, 1]

		# EDM-style: dùng ln(sigma)/4 làm đầu vào thời gian cho mạng
		# Ánh xạ dải [sigma_min, sigma_max] về dải số phù hợp cho sinusoidal embedding
		c_noise = 0.25 * torch.log(sigma.clamp(min=1e-8))

		F_out = self.net(x, c_noise, mess_dropout)
		return c_skip * x + c_out * F_out


class EMAModel:
	"""Quản lý bản sao Exponential Moving Average của model.

	theta^- = mu * theta^- + (1 - mu) * theta
	Mạng EMA làm "điểm tựa" ổn định cho mạng chính học theo.
	"""

	def __init__(self, model, decay=0.999):
		self.decay = decay
		self.ema_model = copy.deepcopy(model)
		self.ema_model.requires_grad_(False)

	def update(self, model):
		"""Cập nhật tham số EMA sau mỗi bước gradient."""
		with torch.no_grad():
			for ema_p, model_p in zip(
				self.ema_model.parameters(), model.parameters()
			):
				ema_p.data.mul_(self.decay).add_(
					model_p.data, alpha=1.0 - self.decay
				)

	def __call__(self, *args, **kwargs):
		return self.ema_model(*args, **kwargs)

	def state_dict(self):
		return self.ema_model.state_dict()

	def load_state_dict(self, state_dict):
		self.ema_model.load_state_dict(state_dict)


class ConsistencyDiffusion(nn.Module):
	"""Consistency Training process thay thế GaussianDiffusion.

	Khác biệt chính so với GaussianDiffusion:
	- Training: dùng Consistency Loss giữa mạng hiện tại và mạng EMA
	- Inference: khử nhiễu 1 bước duy nhất (không cần vòng lặp)
	- Curriculum: N tăng dần từ N_min lên N_max trong quá trình huấn luyện

	Noise model: x_t = x_0 + t * epsilon (EDM-style continuous time)
	"""

	def __init__(self, sigma_min=0.002, sigma_max=1.0, rho=7.0,
				 N_min=10, N_max=80, total_training_steps=1000,
				 loss_type='pseudo_huber', inference_steps=1,
				 inference_sigma=-1):
		super(ConsistencyDiffusion, self).__init__()
		self.sigma_min = sigma_min
		self.sigma_max = sigma_max
		self.rho = rho
		self.N_min = N_min
		self.N_max = N_max
		self.total_training_steps = max(total_training_steps, 1)
		self.loss_type = loss_type
		self.current_step = 0
		self.inference_steps = max(inference_steps, 1)
		# Nếu -1, dùng sigma_max; nếu > 0, dùng giá trị chỉ định
		self.inference_sigma = sigma_max if inference_sigma < 0 else inference_sigma

	def get_N(self):
		"""Curriculum schedule: N tăng từ N_min đến N_max theo căn bậc hai.

		Ban đầu N nhỏ → khoảng cách delta_t lớn → gradient ổn định.
		Sau đó N tăng → delta_t nhỏ → giảm sai số rời rạc hóa.
		"""
		k = min(self.current_step, self.total_training_steps)
		K = self.total_training_steps
		N = math.ceil(math.sqrt(
			(k / K) * (self.N_max ** 2 - self.N_min ** 2) + self.N_min ** 2
		))
		return max(min(N, self.N_max), max(self.N_min, 2))

	def get_timesteps(self, N):
		"""Tạo dãy thời gian rời rạc theo Karras schedule.

		t_i = (sigma_min^(1/rho) + i/(N-1) * (sigma_max^(1/rho) - sigma_min^(1/rho)))^rho

		Phân bố dày hơn ở vùng nhiễu thấp (quan trọng hơn cho chất lượng).
		"""
		if N <= 1:
			return torch.tensor([self.sigma_min], dtype=torch.float32)
		indices = torch.arange(N, dtype=torch.float64)
		t = (
			self.sigma_min ** (1.0 / self.rho)
			+ indices / (N - 1)
			* (self.sigma_max ** (1.0 / self.rho) - self.sigma_min ** (1.0 / self.rho))
		) ** self.rho
		return t.float()

	def _distance(self, x, y):
		"""Hàm khoảng cách d(x, y) cho consistency loss.

		Pseudo-Huber: mượt hơn L2 tại gốc, tránh gradient quá lớn khi hai output gần nhau.
		Returns: loss mỗi sample [batch_size]
		"""
		if self.loss_type == 'pseudo_huber':
			c = 0.00054 * math.sqrt(x.shape[-1])
			return (torch.sqrt((x - y) ** 2 + c ** 2) - c).mean(dim=-1)
		elif self.loss_type == 'l2':
			return ((x - y) ** 2).mean(dim=-1)
		else:  # l1
			return torch.abs(x - y).mean(dim=-1)

	def training_losses(self, model, ema_model, x_start, itmEmbeds,
						batch_index, model_feats):
		"""Tính loss tổng hợp: L_CT + alpha * L_gc.

		Quy trình Consistency Training (Algorithm 3, Song et al. 2023):
		1. Lấy cùng nhiễu z tại hai mốc thời gian liền kề (t_n, t_{n+1})
		2. Mạng hiện tại xử lý tại t_{n+1}, mạng EMA xử lý tại t_n
		3. Loss = khoảng cách giữa hai output (thuộc tính nhất quán)

		Args:
			model: ConsistencyDenoise (tham số theta hiện tại)
			ema_model: EMAModel (tham số theta^- trung bình trượt)
			x_start: vector tương tác gốc [batch_size, num_items]
			itmEmbeds: embedding item cho gc_loss
			batch_index: chỉ số batch
			model_feats: đặc trưng phương thức cho gc_loss

		Returns:
			ct_loss: consistency training loss [batch_size]
			gc_loss: guidance consistency loss [batch_size]
		"""
		batch_size = x_start.size(0)
		device = x_start.device

		N = self.get_N()
		timesteps = self.get_timesteps(N).to(device)

		# Lấy n ngẫu nhiên trong [0, N-2] cho mỗi sample
		n = torch.randint(0, N - 1, (batch_size,), device=device)
		t_n = timesteps[n]       # mức nhiễu thấp hơn
		t_n1 = timesteps[n + 1]  # mức nhiễu cao hơn

		# Dùng CÙNG nhiễu z cho cả hai mốc (key insight của CT)
		z = torch.randn_like(x_start)
		x_t_n1 = x_start + t_n1.unsqueeze(-1) * z  # [batch, items]
		x_t_n = x_start + t_n.unsqueeze(-1) * z    # [batch, items]

		# Mạng hiện tại: dự đoán x_0 từ mức nhiễu cao
		output_theta = model(x_t_n1, t_n1)

		# Mạng EMA (target): dự đoán x_0 từ mức nhiễu thấp (không gradient)
		with torch.no_grad():
			output_ema = ema_model(x_t_n, t_n, mess_dropout=False)

		# Consistency loss: hai output phải giống nhau (thuộc tính nhất quán)
		ct_loss = self._distance(output_theta, output_ema.detach())

		# GC loss: căn chỉnh output mô hình với ground truth trong không gian đặc trưng
		usr_model_embeds = torch.mm(output_theta, model_feats)
		usr_id_embeds = torch.mm(x_start, itmEmbeds)
		gc_loss = ((usr_model_embeds - usr_id_embeds) ** 2).mean(dim=-1)

		self.current_step += 1

		return ct_loss, gc_loss

	def p_sample(self, model, x_start, steps=None, sampling_noise=False):
		"""Inference CM: hỗ trợ 1-step hoặc multi-step.

		Multi-step (K bước):
		1. Thêm nhiễu tại inference_sigma: x_T = x_0 + sigma * noise
		2. Denoise: x_0_hat = f_theta(x_T, sigma)
		3. Re-noise tại mức thấp hơn: x' = x_0_hat + sigma_next * noise'
		4. Denoise lại: x_0_hat = f_theta(x', sigma_next)
		5. Lặp cho đến bước cuối

		Args:
			model: ConsistencyDenoise
			x_start: vector tương tác đầu vào [batch_size, num_items]
			steps: giữ để tương thích API (không dùng)
			sampling_noise: giữ để tương thích API (không dùng)

		Returns:
			output khử nhiễu [batch_size, num_items]
		"""
		batch_size = x_start.size(0)
		device = x_start.device
		K = self.inference_steps

		# Tạo dãy sigma giảm dần cho multi-step
		if K == 1:
			sigmas = [self.inference_sigma]
		else:
			# Chia đều trên Karras schedule từ inference_sigma xuống sigma_min
			ts = self.get_timesteps(K + 1).to(device)
			# Lấy K mốc cao nhất (bỏ sigma_min ở cuối), đảo ngược thành giảm dần
			sigmas = ts.flip(0)[:K].tolist()

		with torch.no_grad():
			# Bước đầu: thêm nhiễu và denoise
			noise = torch.randn_like(x_start)
			sigma_curr = torch.full((batch_size,), sigmas[0], device=device)
			x_t = x_start + sigma_curr.unsqueeze(-1) * noise
			x_0_hat = model(x_t, sigma_curr, mess_dropout=False)

			# Các bước tiếp: re-noise ở mức thấp hơn rồi denoise lại
			for i in range(1, K):
				noise = torch.randn_like(x_start)
				sigma_next = torch.full((batch_size,), sigmas[i], device=device)
				x_t = x_0_hat + sigma_next.unsqueeze(-1) * noise
				x_0_hat = model(x_t, sigma_next, mess_dropout=False)

		return x_0_hat

	def mean_flat(self, tensor):
		return tensor.mean(dim=list(range(1, len(tensor.shape))))

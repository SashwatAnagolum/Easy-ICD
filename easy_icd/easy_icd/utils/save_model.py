import torch

def save_model(model, save_dir, file_name):
	if not os.path.exists(save_dir):
		os.mkdirs(save_dir)

	torch.save(model.state_dict(), os.path.join(save_dir, file_name))
import torch

m_0 = torch.load("checkpoints/qft_2048_eq_4/iter_52533_muls_Decoder.pth.tar")
m_1 = torch.load("checkpoints/qft_2048_eq_4_2/iter_52533_muls_Decoder.pth.tar")
torch.save(m_0, "checkpoints/qft_2048_eq_4_2/iter_52533_muls_Decoder.pth.tar")
print("Over")
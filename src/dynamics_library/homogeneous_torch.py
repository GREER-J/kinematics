import torch


def transx(x: torch.Tensor) -> torch.Tensor:
    I = torch.eye(4, dtype=x.dtype, device=x.device)
    I[0, 3] = x
    return I


def transy(y: torch.Tensor) -> torch.Tensor:
    I = torch.eye(4, dtype=y.dtype, device=y.device)
    I[1, 3] = y
    return I


def transz(z: torch.Tensor) -> torch.Tensor:
    I = torch.eye(4, dtype=z.dtype, device=z.device)
    I[2, 3] = z
    return I


def rotx(phi_rad: torch.Tensor) -> torch.Tensor:
    c = torch.cos(phi_rad)
    s = torch.sin(phi_rad)
    Z = torch.tensor(0.0, dtype=phi_rad.dtype, device=phi_rad.device)
    O = torch.tensor(1.0, dtype=phi_rad.dtype, device=phi_rad.device)

    return torch.stack([
        torch.stack([O, Z, Z, Z]),
        torch.stack([Z, c, -s, Z]),
        torch.stack([Z, s,  c, Z]),
        torch.stack([Z, Z, Z, O])
    ]).squeeze(-1)


def roty_torch(theta_rad: torch.Tensor) -> torch.Tensor:
    c = torch.cos(theta_rad)
    s = torch.sin(theta_rad)
    Z = torch.tensor(0.0, dtype=theta_rad.dtype, device=theta_rad.device)
    O = torch.tensor(1.0, dtype=theta_rad.dtype, device=theta_rad.device)

    return torch.stack([
        torch.stack([ c, Z,  s, Z]),
        torch.stack([ Z, O,  Z, Z]),
        torch.stack([-s, Z,  c, Z]),
        torch.stack([ Z, Z,  Z, O])
    ]).squeeze(-1)


def rotz(psi_rad: torch.Tensor) -> torch.Tensor:
    c = torch.cos(psi_rad)
    s = torch.sin(psi_rad)
    Z = torch.tensor(0.0, dtype=psi_rad.dtype, device=psi_rad.device)
    O = torch.tensor(1.0, dtype=psi_rad.dtype, device=psi_rad.device)

    return torch.stack([
        torch.stack([ c, -s, Z, Z]),
        torch.stack([ s,  c, Z, Z]),
        torch.stack([ Z,  Z, O, Z]),
        torch.stack([ Z,  Z, Z, O])
    ]).squeeze(-1)

import torch
import torch.nn.functional as F


def batched_matrix_multiply(anchor_rep, non_anchor_shuff, chunk_size):
    bsz, dim = non_anchor_shuff.shape
    logits = torch.zeros((bsz, bsz), device=anchor_rep.device, dtype=torch.float16)

    for i in range(0, bsz, chunk_size):
        chunk = non_anchor_shuff[i:i + chunk_size]
        logits[:, i:i + chunk_size] = anchor_rep @ chunk.T

    return logits


class Total_Correlation:
    def __init__(self, negative_sampling: str = "n"):
        self.negative_sampling = negative_sampling

    def compute_logits_n(self, anchor_rep, non_anchor_reps):
        non_anchor_shuff = torch.ones_like(anchor_rep)
        for r in non_anchor_reps:
            non_anchor_shuff = non_anchor_shuff * r[torch.randperm(r.shape[0])]

        logits = anchor_rep @ torch.t(non_anchor_shuff) # (bsz, bsz)

        MIP_of_positive_samples = anchor_rep.clone()
        for r in non_anchor_reps:
            MIP_of_positive_samples = MIP_of_positive_samples * r
        MIP_of_positive_samples = MIP_of_positive_samples.sum(axis=1) # (bsz)

        return torch.where(torch.eye(n=anchor_rep.shape[0]).to(anchor_rep.device) > 0.5,
                           MIP_of_positive_samples,
                           logits)

    def compute_non_anchor_products(self, tensors):
        if len(tensors) == 2:
            y, z = tensors
            y_z = []
            for i in range(y.shape[0]):
                y_z.append(y * z)
                z = torch.roll(z, shifts=1, dims=0)
            return y_z

        x = tensors[0]

        partial_products = self.compute_non_anchor_products(tensors[1:])

        all_products = []
        for i in range(x.shape[0]):
            for partial_product in partial_products:
                all_products.append(partial_product * x)
            x = torch.roll(x, shifts=1, dims=0)

        return all_products

    def compute_logits_n_squared(self, anchor_rep, non_anchor_reps):
        
        non_anchor_products = self.compute_non_anchor_products(non_anchor_reps)
        non_anchor_product = torch.cat(non_anchor_products, 0)
        logits = anchor_rep @ non_anchor_product.T

        return logits

    def forward(self, representations, logit_scale):
        labels = torch.arange(representations[0].shape[0]).to(representations[0].device)
        losses = []

        for i, r in enumerate(representations):
            if self.negative_sampling == "n":
                logits = logit_scale * self.compute_logits_n(r, [rep for j, rep in enumerate(representations) if i != j])
            elif self.negative_sampling == "n_squared":
                logits = logit_scale * self.compute_logits_n_squared(r, [rep for j, rep in enumerate(representations) if i != j])
            else:
                raise ValueError("Invalid value for negative_sampling. Expected 'n' or 'n_squared'.")

            loss = F.cross_entropy(logits, labels)

            losses.append(loss)

        return sum(losses) / len(losses)

    def __call__(self, representations, logit_scale):
        return self.forward(representations, logit_scale)

# Rankup

RankUp is a PyTorch extension that nudges gradient updates away from directions that would lower the rank of your model’s parameters or features. It can be useful if you need to preserve full-rank behavior, for instance in certain regularization or constraint settings.

It calculates the gradient of the “smooth rank” (∇R) with respect to x.
If the inner product (∇R ⋅ ∇L) is positive—indicating the loss gradient is decreasing rank—it removes that component from the final backward pass.

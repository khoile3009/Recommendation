from src.processing.get_rating_matrix import get_sparse_matrix

num_user = 1000
num_item = 1000

sparse_matrix = get_sparse_matrix(num_user, num_item)
sparsity = len(sparse_matrix.nonzero()[0])
sparsity /= num_item * num_user
print(sparsity)


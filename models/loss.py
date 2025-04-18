import torch

def multilabel_categorical_crossentropy(y_true, y_pred):
    loss_mask = y_true != -100
    y_true = y_true.masked_select(loss_mask).view(-1, y_pred.size(-1)) 
    y_pred = y_pred.masked_select(loss_mask).view(-1, y_true.size(-1)) 
    y_pred = (1 - 2 * y_true) * y_pred 
    y_pred_neg = y_pred - y_true * 1e12
    y_pred_pos = y_pred - (1 - y_true) * 1e12
    zeros = torch.zeros_like(y_pred[:, :1])
    y_pred_neg = torch.cat([y_pred_neg, zeros], dim=-1)
    y_pred_pos = torch.cat([y_pred_pos, zeros], dim=-1)
    neg_loss = torch.logsumexp(y_pred_neg, dim=-1) 
    pos_loss = torch.logsumexp(y_pred_pos, dim=-1) 
    return (neg_loss + pos_loss).mean() 

def global_singular_smoothing_loss(label_weight):
    loss = -torch.norm(label_weight, p='nuc') / torch.sum(torch.norm(label_weight, dim=1))
    return loss

def local_singular_smoothing_loss(label_weight, depth2label, num_class):
    hie_loss = []
    loss = 0
    for depth in depth2label:
        if len(depth2label[depth]) > 1 :
            hie_loss.append((depth + 1) * (len(depth2label[depth]) / num_class) * (-torch.norm(label_weight[depth2label[depth]], p='nuc') / torch.sum(torch.norm(label_weight[depth2label[depth]], dim=1))))
    for l in hie_loss:
        loss += l
    return loss

class TreeNode:
    def __init__(self, name, id):
        self.name = name.split('/')[-1]
        self.id = id
        self.depth = None
        self.embedding = None 
        self.children = []
        self.parent = None
    
    def add_child(self, child):
        self.children.append(child)

class structural_entropy_based_loss():
    def __init__(self, taxonomy_list, value_dict):
        super(structural_entropy_based_loss, self).__init__()
        self.taxonomy_list = taxonomy_list
        self.value_dict = value_dict
        self.labelnodes_dict = {}
        self.root = self.build_tree()
        self.tree_height = self.compute_tree_height(self.root)
    
    def build_tree(self):
        root = None
        for line in self.taxonomy_list:
            parent_name = line[0]
            if self.value_dict[parent_name] not in self.labelnodes_dict:
                self.labelnodes_dict[self.value_dict[parent_name]] = TreeNode(parent_name, self.value_dict[parent_name])
            parent_node = self.labelnodes_dict[self.value_dict[parent_name]]
            if root is None:
                root = parent_node
                parent_node.depth = 0
                parent_node.id = -1
                parent_node.parent = -1

            for child_name in line[1:]:
                if self.value_dict[child_name] not in self.labelnodes_dict:
                    self.labelnodes_dict[self.value_dict[child_name]] = TreeNode(child_name, self.value_dict[child_name])
                    self.labelnodes_dict[self.value_dict[child_name]].depth = parent_node.depth + 1
                    self.labelnodes_dict[self.value_dict[child_name]].parent = parent_node.id
                child_node = self.labelnodes_dict[self.value_dict[child_name]]
                parent_node.add_child(child_node)
        return root
    
    def compute_tree_height(self, node):
        if not node.children:
            return 0
        return 1 + max(self.compute_tree_height(child) for child in node.children)
    
    def NPSI_loss(self, A: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        """
        Compute the NPSI (Normalized Pointwise Structural Information) loss.
        
        Args:
            A (torch.Tensor): Adjacency matrix (n x n).
            Y (torch.Tensor): Node embeddings (n x r).
            
        Returns:
            torch.Tensor: The NPSI loss value.
        """
        # Ensure A and Y are tensors
        Y = torch.tensor(Y, dtype=torch.float32)

        # Calculate AY
        AY = torch.matmul(A, Y)

        # Create a matrix of ones with the same shape as Y^T * AY
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        ones_matrix = torch.ones((Y.shape[1], Y.shape[0])).to(device)
        
        # Calculate Y^T * A * Y
        YT1AY = torch.matmul(Y.T - ones_matrix, torch.matmul(A, Y))
        
        # Calculate sum(A)
        normalization_factor = torch.sum(A)
        
        # Normalize YTAY
        normalized_YT1AY = YT1AY / normalization_factor

        # Calculate the normalized term
        normalized_AY = torch.matmul(ones_matrix, AY) / normalization_factor
        
        # Calculate the logarithm
        log_term = torch.log2(normalized_AY)
        
        # Calculate the element-wise product
        elementwise_product = torch.mul(normalized_YT1AY, log_term)
        
        # Calculate the trace
        npsi_loss = torch.trace(elementwise_product)
        
        return npsi_loss
    
    def level_agg(self, A, Y):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        Y = torch.where(Y == -100, torch.tensor(0).to(device), Y)
        # Ensure A and Y are tensors
        Y = torch.tensor(Y, dtype=torch.float32)

        class_counts = Y.sum(dim=0)  
        class_sums = torch.matmul(Y.T, A)  
        class_counts[class_counts == 0] = 1
        class_means = class_sums / class_counts.unsqueeze(1)
        non_zero_mask = class_means.abs().sum(dim=1) != 0
        non_zero_rows = class_means[non_zero_mask]
        non_zero_indices = torch.nonzero(non_zero_mask, as_tuple=True)[0]
        non_zero_indices_list = non_zero_indices.tolist()

        parent_indices_list = []
        for child_id in non_zero_indices_list:
            parent_id = self.labelnodes_dict[child_id].parent
            parent_indices_list.append(parent_id)

        num_non_zero_rows = non_zero_rows.shape[0]
        fixed_columns = Y.shape[1]
        Y_agg = torch.zeros((num_non_zero_rows, fixed_columns), dtype=Y.dtype).to(device)

        for i, idx in enumerate(parent_indices_list):
            if idx != -1:
                Y_agg[i, idx] = 1

        return non_zero_rows, Y_agg
    
    def forward(self, outputs, labels):
        """
        outputs:(batch, labels level, 768)
        label:(batch, labels num * labels level)

        text_embedding:(labels level, batch, 768) 
        text_embedding[i]:i level text embedding

        Y_total:(labels level, batch, label num) 
        Y: (batch, level label num)

        """
        text_embedding = outputs.permute(1, 0, 2)
        Y_total = labels.view(labels.shape[0], text_embedding.shape[0], -1)
        Y_total = Y_total.permute(1, 0, 2)
        level_label = []
        for l in range(text_embedding.shape[0] + 1):
            level_label.append([])
        for label_nodes in self.labelnodes_dict.values():
            level_label[label_nodes.depth].append(label_nodes.id)

        total_npsi_loss = 0
            
        agg_embedding = None
        agg_Y = None

        Y = Y_total[text_embedding.shape[0] - 1]
        A = text_embedding[text_embedding.shape[0] - 1]
        agg_embedding, agg_Y = self.level_agg(A, Y)

        Y = Y[:, level_label[text_embedding.shape[0]]]
        non_zero_rows = torch.any(Y != 0, dim=1)
        Y = Y[non_zero_rows]
        A = A[non_zero_rows]
        
        non_zero_cols = torch.any(Y != 0, dim=0)
        Y_filtered = Y[:, non_zero_cols]

        A = torch.matmul(A, A.T)
        A_sigmoid = torch.sigmoid(A) 
        if Y_filtered.shape[0] > 0 and A_sigmoid.shape[0] > 0:
            total_npsi_loss += self.NPSI_loss(A_sigmoid, Y_filtered)

        for i in reversed(range(text_embedding.shape[0] - 1)): 
            Y = Y_total[i]
            A = text_embedding[i]
            A = torch.cat((A, agg_embedding), dim=0)
            Y = torch.cat((Y, agg_Y), dim=0)
            agg_embedding, agg_Y = self.level_agg(A, Y)

            Y = Y[:, level_label[i + 1]]
            non_zero_rows = torch.any(Y != 0, dim=1)
            Y = Y[non_zero_rows]
            A = A[non_zero_rows]

            non_zero_cols = torch.any(Y != 0, dim=0)
            Y_filtered = Y[:, non_zero_cols]

            A = torch.matmul(A, A.T)
            A_sigmoid = torch.sigmoid(A) 
            if Y_filtered.shape[0] > 0 and A_sigmoid.shape[0] > 0:
                total_npsi_loss += self.NPSI_loss(A_sigmoid, Y_filtered)




        return total_npsi_loss

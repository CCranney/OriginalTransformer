from torch import nn

class MultiHeadedAttention(nn.Module):
    
    """
    The multi-head attention layer.
    
    This class effectively performs the attention function on the input multiple times, then returns a consolidated
        output. See attention.py for (many) more details as to the purpose of this function.
        
    If you followed the https://jalammar.github.io/illustrated-transformer/ blog, you should have a grasp of how
        attention works in general. What it did not prepare you for, however, was the VECTORIZED version of applying
        multi head attention. This makes everything worlds faster, but makes step-by-step interpretation code 
        readthrough not directly applicable to what you read in the blog. Here is a breakdown of what happens.
        
    1. You provide an input. In the case of this transformer, each input is a batch of sentences. These sentences are 
        represented by the following dimensions:
        
        [batch_size, num_tokens, model_size/word_vector_length]
        
        or, in plain english:
        
        [# of sentences in the batch, # of words in the longest sentence, length of word vector per word]
        
        Shorter sentences are padded. More an that in a future file. Let's say, as an example moving forward, we
            have a batch of 32 sentences, the longest of which is 11 words long, and the word vector size is 512.
            So the input has a dimension of [32, 11, 512].
            
    2. Initialize the query_weight, key_weight, and value_weight weight matrices. This is where the illustrated  
        intuition will begin to fail. 
        
        There is a q, k, and v weight for each head. When you initialize them, however, you're going to cheat a 
        bit. Instead of making each weight matrix for each head separately, you can just initialize the q, k, and v
        matrices all in a single go. In this case, you make 3 [model_size, model_size] weight matrices. The model size
        MUST be divisible by the number of heads, because this is really each head stacked on top of each other in a 
        real graph. So while it looks like [model_size, model_size], it's practically the same as 
        [number of heads, head size, model size]. In our case, model_size is 512, so you have query, key, and value
        weight matrices of size:
        
        [model_size, model_size]
        
        which in our example is
        
        [512, 512]
        
        If we have 8 heads, that means each head dimension is 64. Realistically each of these weight matrices is really
            [8, 64, 512]. See the next number for why we left them as a giant 2D array.
            
        NOTE: steps 3-5 are all done in a single line of code. Look under the `forward` function for the convoluted
            list comprehension line.
            
    3. Multiply the input by each of the query_weight, key_weight, and value_weight matrices. This is why we left 
        them in a square [model_size, model_size] format - because of how matrix multiplication works, multiplying 
        the input by one of these is the same as multiplying it by each one individually, but way faster/more 
        efficient.
            
            The output of this multiplication, because you're multiplying it by a square, is the same dimension as the
            input - [batch_size, num_words, model_size]. You just multiplied all the heads by the input at once. So in
            our example we now have:
            [32, 11, 512]
            for query, key, and value each.
            
    4. So you now have you query, key, value values ready for the attention algorithm. Now, however, we actually
        DO want each of the heads to be separated out from each other - each head should be run through the
        algorithm separately. So we take our outputs and break that final dimension down from [model_size] into
            [number of heads, head dimension size]. In our example, we are breaking down the following:
            
            [32, 11, 512]
            
            Into the following:
            
            [32, 11, 8, 64]
            
    5. This is almost right, but bear in mind that we want the number of words dimension to be by the query, key, 
        and value representation dimension. In other words, the attention algorithm is expecting graphs of shape
        [number of words, head dimension size]. So to make this so, we need to swap the 2nd and 3rd dimensions into:
            
            [32, 8, 11, 64]
            [batch size, number of heads, number of words, head dimension size]
            
            NOW you are back on track for the illustrated transformer blog explanation!
            
    6. The 4D query, key, and value matrices are passed into the attention algorithm. The output is the same shape
        as the input:
            
            [32, 8, 11, 64]
            
    7. We need to undo steps 5 and 4, in that order. We are basically going to do our best to reshape the encoded
        representations to be back to the same shape as the original input.
            
            So swap 2nd and 3rd dimension again:
            
            [32, 11, 8, 64]
            
            Then reshape back into the model shape:
            
            [32, 11, 512]
            
    8. We multiply this encoded representation matrix by an output weight matrix. This condenses the output of the 
        different heads into a single cohesize output of the multi-head attention class. This output weight matrix
        is the same shape as the full query_weights, key_weights, and value_weights defined in step 2 (and is often 
        made at the same time). The multiplication output is the same as the output of step 3 - which is to say, it's 
        the same shape as the original input.

            [32, 11, 512]

    Vectorization, man. It's a different way of thinking entirely.
    """
    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        "Implements Figure 2"
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = [
            lin(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
            for lin, x in zip(self.linears, (query, key, value))
        ]

        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = attention(
            query, key, value, mask=mask, dropout=self.dropout
        )

        # 3) "Concat" using a view and apply a final linear.
        x = (
            x.transpose(1, 2)
            .contiguous()
            .view(nbatches, -1, self.h * self.d_k)
        )
        del query
        del key
        del value
        return self.linears[-1](x)

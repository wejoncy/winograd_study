import numpy as np
import time

#generate by  https://github.com/andravin/wincnn
G_F23 = np.array([
     [ 1.0,  0.0, 0.0 ],
     [ 0.5,  0.5, 0.5 ],
     [ 0.5, -0.5, 0.5 ],
     [ 0.0,  0.0, 1.0 ]])
Bt_F23 = np.array([
     [ 1.0,  0.0, -1.0,  0.0 ],
     [ 0.0,  1.0,  1.0,  0.0 ],
     [ 0.0, -1.0,  1.0,  0.0 ],
     [ 0.0,  1.0,  0.0, -1.0 ]])
At_F23 = np.array([
     [ 1.0, 1.0,  1.0,  0.0 ],
     [0.0, 1.0, -1.0, -1.0]])
     
# statictic time cost for direct conv and wino conv
def timing(f):
    def inner(x, w):
        start = time.time()
        time.sleep(0.12)
        y = f(x, w)
        end = time.time()
        print('time：%s' % (end - start))
        return y

    return inner

# Cal matrixs U
def trans_kernel(g):
    return np.dot(np.dot(G_F23, g), G_F23.T)

# Cal matrixs V
def trans_input(d):
    return np.dot(np.dot(Bt_F23, d), Bt_F23.T)
    
# Cal matrixs M
def trans_output(r):
     return np.dot(np.dot(At_F23,r),At_F23.T)

# Cal matrixs Y
def wino_f23(kernel,input):
     tran_inp = trans_input(input)
     tran_ker = trans_kernel(kernel)
     mid = tran_inp * tran_ker
     out = trans_output(mid)
     return out

#tiling and call wino_f23
@timing
def conv_direct(kernel,input):
    out=np.zeros(input.shape)
    for h in range(input.shape[0]):
        for w in range(input.shape[1]):
            if h+3>=input.shape[0] or w+3>=input.shape[0]:continue
            out[h,w]=np.sum(input[h:h+3,w:w+3]*kernel)
    return out

#used for calculate the correct answer
@timing
def winograd(kernel,input):
    HH=input.shape[0]
    outwin=np.zeros(input.shape)
    for i in range(0,HH//2*4,2):
        for j in range(0,HH//2*4,2):
            t_i = input[i:i+4,j:j+4]
            if t_i.shape != (4,4):continue
            out_wino = wino_f23(kernel,t_i)
            outwin[i:i+2,j:j+2]=out_wino
    return outwin


np.set_printoptions(threshold=np.inf)
def test():
    input=np.array([
        [0,1,2,3],
        [4,5,6,7],
        [8,9,10,11],
        [12,13,14,15]
    ])
    HH=1033
    input=np.arange(HH*HH).reshape(HH,HH)
    kernel=np.array([
    [1,2,1],
    [2,1,0],
    [1,1,2]
    ])

    outwin=winograd(kernel,input)        
    print(outwin[0:1,1:10])
    out_direct= conv_direct(kernel,input)
    print("out_direct:\n",out_direct[0:1,1:10])
    #print("max error: ",np.max(np.abs(out_wino-out_direct)))
    np.testing.assert_allclose(outwin,out_direct)
    print("chech passed")

if __name__ == "__main__":
    test()
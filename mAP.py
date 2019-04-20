def calmAP(output, target):
    '''
    Args:
        output: preds of model softmax need to be used.
        target: gt
    Return:
        cAP: 
    '''
    ap = np.zeros(output.shape[1])
    for k in range(output.shape[1]):
        outputk = output[:, k]
        targetk = target[:, k]
        sortind = np.argsort(-outputk, axis=0)
        apsum = 0.0
        right = 0.0
        for i in range(len(sortind)):
            idx = sortind[i]
            if targetk[idx] == 1:
                right = right + 1
                apsum = apsum + right/(i+1)
        ap[k] = apsum/sum(targetk)
    return mAP

def calcAP(output, target, w=4):
    '''
    Args:
        output: preds of model softmax need to be used.
        target: gt
        w: weight
    Return:
        cAP: 
    '''
    ap = np.zeros(output.shape[1])
    for k in range(output.shape[1]):
        outputk = output[:, k]
        targetk = target[:, k] * w
    #     print(targetk)
        sortind = np.argsort(-outputk, axis=0)
        apsum = 0.0
        right = 0.0
        for i in range(len(sortind)):
            idx = sortind[i]
            if targetk[idx] == w:
                right = right + 1*w
                addw = lambda w: (w-1)*right/w if w!=1 else 0
                apsum = apsum + right/(i+1+ addw(w))
        ap[k] = apsum/sum(targetk)
        
    return ap.mean()

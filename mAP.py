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

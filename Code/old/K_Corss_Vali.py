def div_list(ls,n):
    '''
    '''
    if not isinstance(ls,list) or not isinstance(n,int):
        return []
    ls_len = len(ls)
    if n<=0 or 0==ls_len:
        return []
    if n > ls_len:
        return []
    elif n == ls_len:
        return [[i] for i in ls]
    else:
        j = ls_len//n
        k = ls_len%n
        ls_return = []
        for i in range(0,(n-1)*j,j):
            ls_return.append(ls[i:i+j])
        ls_return.append(ls[(n-1)*j:])
        return ls_return

def K_Cross_Split(path, K):
    filelist = os.listdir(path)
    high = []
    low = []
    for onefile in filelist:
        if 'high' in onefile:
            high.append(onefile)
        else:
            low.append(onefile)
    # print(len(filelist))
    # print(len(high))
    # print(len(low))

    random.shuffle(high)
    random.shuffle(low)

    divhigh = div_list(high, K)
    divlow = div_list(low, K)

    res = []
    for i in range(0, K):
        temp = divhigh[i] + divlow[i]
        random.shuffle(temp)
        res.append(temp)

    return res

def get_train_and_test_filenames(K, m, K_folders):
    train_data = []
    test_data = []
    test_data.append(K_folders[m])
    for i in range(0, K):
        if i != m:
            train_data.append(K_folders[i])
    return train_data, test_data
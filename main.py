from utils import * 
        
    
if __name__ == "__main__":
    M = 2
    nMLD = 10
    nSLD = [10, 10]
    tt = [36, 36]
    tf = [28, 28]
    # mld_lambda = 0.02 # max: 0.05*2 = 0.1
    sld_lambda = 0.001 # arrival rate per slot
    W = 16
    K = 6
    kk = [0.5, 0.5]
    mld_lambda_checklist = np.arange(0.002, 0.0031, 0.0005) ## arrival rate per slot
    throughput_list = []
    access_delay_list = []
    queuing_delay_list = []
    e2e_delay_list = []
    # change mld_lambda:
    for mld_lambda in mld_lambda_checklist:
        print(f"mld_lambda: {mld_lambda}:")
        ac_dl = 0
        q_dl = 0
        e2e_dl = 0
        thrpt = 0
        for i in range(M):
            # uu
            p, throughput, uu = calc_uu_p(nMLD, mld_lambda * kk[i], nSLD[i], sld_lambda, tt[i], tf[i], W, K, W, K)
            # _, ad = calc_access_delay(p, tt[i], tf[i], W, K, mld_lambda * kk[i] ) # MLD delay  
            
            print("uu throughput ", throughput/tt[i])
            if uu:
                status = get_status(p, tt[i], tf[i], W, K, mld_lambda * kk[i]) + get_status(p, tt[i], tf[i], W, K, sld_lambda )
                if status == "UU":
                    # print(p, throughput)
                    qd, ad = calc_access_delay(p, tt[i], tf[i], W, K, mld_lambda * kk[i] ) # MLD delay
                    qd_sld, ad_sld = calc_access_delay(p, tt[i], tf[i], W, K, sld_lambda ) # SLD delay
                    print(f"  {i}:", f"status: UU, queuing-delay: {qd}, access-delay: {ad}, p: {p}, throughput: {throughput/tt[i]}, input-rate: {nMLD * mld_lambda * kk[i] + nSLD[i] * sld_lambda}")
                    ac_dl += kk[i] * ad
                    q_dl += kk[i] * qd
                    thrpt += throughput / tt[i]
                    continue
            # su
            p, throughput, su = calc_su_p(nMLD, mld_lambda*kk[i], nSLD[i], sld_lambda, tt[i], tf[i], W, K, W, K)
            print("su throughput ", throughput/tt[i])
            
            if su:
                qd, ad = calc_access_delay(p, tt[i], tf[i], W, K, mld_lambda * kk[i]) # MLD delay
                qd_sld, ad_sld = calc_access_delay(p, tt[i], tf[i], W, K, sld_lambda) # SLD delay
                # print(f"  {i}:", "SU", qd, ad, qd_sld, ad_sld)
                print(f"  {i}:", f"status: SU, queuing-delay: {qd}, access-delay: {ad}, p: {p}, throughput: {throughput/tt[i]}, input-rate: {nMLD * mld_lambda * kk[i] + nSLD[i] * sld_lambda}")
                ac_dl += kk[i] * ad
                q_dl += kk[i] * qd
                thrpt += throughput / tt[i]
                continue
            # us
            p, throughput, us = calc_us_p(nMLD, mld_lambda*kk[i], nSLD[i], sld_lambda, tt[i], tf[i], W, K, W, K)
            print("us throughput ", throughput/tt[i])
            
            if us:
                qd, ad = calc_access_delay(p, tt[i], tf[i], W, K, mld_lambda * kk[i]) # MLD delay
                qd_sld, ad_sld = calc_access_delay(p, tt[i], tf[i], W, K, sld_lambda) # SLD delay
                # print(f"  {i}:", "US", qd, ad, qd_sld, ad_sld)
                print(f"  {i}:", f"status: US, queuing-delay: {qd}, access-delay: {ad}, p: {p}, throughput: {throughput/tt[i]}, input-rate: {nMLD * mld_lambda * kk[i] + nSLD[i] * sld_lambda}")
                ac_dl += kk[i] * ad
                q_dl += kk[i] * qd
                thrpt += throughput / tt[i]
                continue
            # ss
            p, throughput, ss = calc_ss_p(nMLD, mld_lambda*kk[i], nSLD[i], sld_lambda, tt[i], tf[i], W, K, W, K)
            print("ss throughput ", throughput/tt[i])
            
            if ss:
                qd, ad = calc_access_delay(p, tt[i], tf[i], W, K, mld_lambda * kk[i]) # MLD delay
                qd_sld, ad_sld = calc_access_delay(p, tt[i], tf[i], W, K, sld_lambda) # SLD delay
                # print(f"link{i}", "SS", qd, ad, qd_sld, ad_sld)
                print(f"  {i}:", f"status: SS, queuing-delay: {qd}, access-delay: {ad}, p: {p}, throughput: {throughput / tt[i]}, input-rate: {nMLD * mld_lambda * kk[i] + nSLD[i] * sld_lambda}")
                ac_dl += kk[i] * ad
                q_dl += kk[i] * qd
                thrpt += throughput / tt[i]
                continue
            print("no status")
        e2e_dl = ac_dl + q_dl
        access_delay_list.append(ac_dl)
        queuing_delay_list.append(q_dl)
        e2e_delay_list.append(e2e_dl)
        throughput_list.append(thrpt)
    print("result:")
    print(f"acc delay: {access_delay_list}")
    print(f"queuing delay: {queuing_delay_list}")
    print(f"e2e delay: {e2e_delay_list}")
    print(f"throughput: {throughput_list}")
        

    
    
    
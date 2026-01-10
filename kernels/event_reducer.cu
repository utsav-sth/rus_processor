#include <stdio.h>
#include <assert.h>
#include <math.h>

#define MAX_HITS_PER_EVENT 1000
#define MAX_EVENTS 10000
#define MAX_FEATURES 6
#define ClusterSizeMax 100
#define nChamberPlanes 30
#define DEBUG_ER 1

__device__ bool is_chamber(int d){return d >= 1 && d <= 30;}
__device__ bool is_proptube(int d){return d >= 31 && d <= 46;}
__device__ bool is_hodoscope(int d){return d >= 47 && d <= 54;}

__device__ int event_reduction(float hit_data[MAX_HITS_PER_EVENT][MAX_FEATURES], short* hitflag, int nhits, int detid){
    printf("ER kernel start\n");
    float w_max, w_min, dt_mean;
    int cluster_iAH_arr_cur, cluster_iAH_arr_size;
    int cluster_iAH_arr[ClusterSizeMax];
    int uniqueID, uniqueID_curr;
    float tdcTime_curr;
    int iAH;
    int nAH_reduced = 0;
#if DEBUG_ER
    int nDiscard_flag = 0;
    int nDiscard_overlap = 0;
#endif
    cluster_iAH_arr_size = 0;
    uniqueID = uniqueID_curr = -1;
    bool chanhashit[202] = {0};
    for(iAH = 0; iAH < nhits; ++iAH){
        hitflag[iAH] = 1;
        float chan = hit_data[iAH][0];
        float tdc = hit_data[iAH][3];
        float drift = hit_data[iAH][4];
        if(((int)hit_data[iAH][5] & (1 << 1)) == 0){
            hitflag[iAH] = -1;
#if DEBUG_ER
            ++nDiscard_flag;
#endif
            continue;
        }
        if(detid < 31 || detid > 46){
            uniqueID = detid * 1000 + (int)chan;
            uniqueID_curr = -1;
            if(uniqueID != uniqueID_curr){
                uniqueID_curr = uniqueID;
                tdcTime_curr = tdc;
            }else{
                if(detid > 36 || fabsf(tdc - tdcTime_curr) < 80.f){
                    hitflag[iAH] = -1;
#if DEBUG_ER
                    ++nDiscard_overlap;
#endif
                    continue;
                }else{
                    tdcTime_curr = tdc;
                }
            }
        }
        if(detid <= nChamberPlanes){
            if(cluster_iAH_arr_size == 0){
                cluster_iAH_arr[0] = iAH;
                ++cluster_iAH_arr_size;
            }else{
                if(fabs(chan - hit_data[cluster_iAH_arr[cluster_iAH_arr_size - 1]][0]) > 1 || iAH == nhits - 1){
                    if(iAH == nhits - 1 && fabs(chan - hit_data[cluster_iAH_arr[cluster_iAH_arr_size - 1]][0]) <= 1){
                        cluster_iAH_arr[cluster_iAH_arr_size] = iAH;
                        ++cluster_iAH_arr_size;
                    }
                    if(cluster_iAH_arr_size == 2){
                        w_max = 0.9f * 0.5f * fabs(hit_data[cluster_iAH_arr[cluster_iAH_arr_size - 1]][2] - hit_data[cluster_iAH_arr[0]][2]);
                        w_min = 4.0f / 9.0f * w_max;
                        if((hit_data[cluster_iAH_arr[0]][4] > w_max && hit_data[cluster_iAH_arr[cluster_iAH_arr_size - 1]][4] > w_min) || (hit_data[cluster_iAH_arr[0]][4] > w_min && hit_data[cluster_iAH_arr[cluster_iAH_arr_size - 1]][4] > w_max)){
                            if(hit_data[cluster_iAH_arr[0]][4] > hit_data[cluster_iAH_arr[cluster_iAH_arr_size - 1]][4]){
                                hitflag[cluster_iAH_arr[0]] = -1;
                            }else{
                                hitflag[cluster_iAH_arr[cluster_iAH_arr_size - 1]] = -1;
                            }
                        }else if(fabs(hit_data[cluster_iAH_arr[0]][3] - hit_data[cluster_iAH_arr[cluster_iAH_arr_size - 1]][3]) < 8.0f && detid >= 19 && detid <= 24){
                            hitflag[cluster_iAH_arr[0]] = -1;
                            hitflag[cluster_iAH_arr[cluster_iAH_arr_size - 1]] = -1;
                        }
                    }else if(cluster_iAH_arr_size >= 3){
                        dt_mean = 0.0f;
                        for(cluster_iAH_arr_cur = 1; cluster_iAH_arr_cur < cluster_iAH_arr_size; ++cluster_iAH_arr_cur){
                            dt_mean += fabs(hit_data[cluster_iAH_arr[cluster_iAH_arr_cur]][3] - hit_data[cluster_iAH_arr[cluster_iAH_arr_cur - 1]][3]);
                        }
                        dt_mean /= (cluster_iAH_arr_size - 1);
                        if(dt_mean < 10.0f){
                            for(cluster_iAH_arr_cur = 0; cluster_iAH_arr_cur < cluster_iAH_arr_size; ++cluster_iAH_arr_cur){
                                hitflag[cluster_iAH_arr[cluster_iAH_arr_cur]] = -1;
                            }
                        }else{
                            for(cluster_iAH_arr_cur = 1; cluster_iAH_arr_cur < cluster_iAH_arr_size; ++cluster_iAH_arr_cur){
                                hitflag[cluster_iAH_arr[cluster_iAH_arr_cur]] = -1;
                            }
                        }
                    }
                    cluster_iAH_arr_size = 0;
                }else{
                    cluster_iAH_arr[cluster_iAH_arr_size] = iAH;
                    ++cluster_iAH_arr_size;
                }
            }
        }
    }
    for(iAH = 0; iAH < nhits; ++iAH){
        int chan = (int)hit_data[iAH][0];
        if(hitflag[iAH] > 0){
            if(chanhashit[chan]){
                hitflag[iAH] = -1;
            }else{
                chanhashit[chan] = true;
                ++nAH_reduced;
            }
        }
    }
#if DEBUG_ER
    if(threadIdx.x == 0){
        printf("[ER] detID=%2d nhits=%3d kept=%3d flagCut=%2d overlap=%2d\n", detid, nhits, nAH_reduced, nDiscard_flag, nDiscard_overlap);
    }
#endif
    return nAH_reduced;
}

__global__ void gkernel_event_reduce(float hit_data[MAX_EVENTS][MAX_HITS_PER_EVENT][MAX_FEATURES], int hit_flags[MAX_EVENTS][MAX_HITS_PER_EVENT], bool* hastoomanyhits, int* datasizes){
    const int evt = blockIdx.x;
    const int nhits = datasizes[evt];
    if(nhits <= 0) return;
    short local_flags[MAX_HITS_PER_EVENT];
    for(int i = 0; i < nhits; ++i) local_flags[i] = 1;
    for(int detid = 1; detid <= 64; ++detid){
        int indices[MAX_HITS_PER_EVENT], group_size = 0;
        for(int i = 0; i < nhits; ++i) if((int)hit_data[evt][i][1] == detid) indices[group_size++] = i;
        if(group_size == 0) continue;
        float sub_hits[MAX_HITS_PER_EVENT][MAX_FEATURES];
        short sub_flags[MAX_HITS_PER_EVENT];
        for(int i = 0; i < group_size; ++i) for(int f = 0; f < MAX_FEATURES; ++f) sub_hits[i][f] = hit_data[evt][indices[i]][f];
        event_reduction(sub_hits, sub_flags, group_size, detid);
        for(int i = 0; i < group_size; ++i) if(sub_flags[i] <= 0) local_flags[indices[i]] = -1;
    }
    int valid_count = 0;
    for(int i = 0; i < nhits; ++i){
        hit_flags[evt][i] = local_flags[i];
        if(local_flags[i] > 0) ++valid_count;
    }
    hastoomanyhits[evt] = (valid_count < 0.5f * nhits);
#if DEBUG_ER
    if(threadIdx.x == 0){
        printf("[ER-EVENT] evt=%4d total=%3d kept=%3d tooMany=%d\n", evt, nhits, valid_count, (int)hastoomanyhits[evt]);
    }
#endif
}

extern "C" void launch_event_reducer(float hit_data[MAX_EVENTS][MAX_HITS_PER_EVENT][MAX_FEATURES], int hit_flags[MAX_EVENTS][MAX_HITS_PER_EVENT], bool* hastoomanyhits, int* datasizes){
    int num_events = datasizes[0];
    cudaDeviceSetLimit(cudaLimitPrintfFifoSize, 8 * 1024 * 1024);
    gkernel_event_reduce<<<num_events, 1>>>(hit_data, hit_flags, hastoomanyhits, datasizes);
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if(err != cudaSuccess){
        printf("[ER-HOST] kernel launch failed: %s\n", cudaGetErrorString(err));
    }else{
        printf("[ER-HOST] kernel completed OK\n");
    }
}

#include "winograd.h"
#include <cblas.h>
#include <arm_neon.h>
#include <string.h>


#define BH 8
#define BW 8
#define BK 32

// Transformation matrices for Winograd F(2x2, 3x3)
const float G[4][3] = {
    {1.0, 0.0, 0.0}, 
    {0.5, 0.5, 0.5}, 
    {0.5, -0.5, 0.5}, 
    {0.0, 0.0, 1.0}
};
const float G_T[3][4] = {
    {1, 0.5, 0.5, 0.0}, 
    {0.0, 0.5, -0.5, 0.0}, 
    {0.0, 0.5, 0.5, 1.0}
};
const float B[4][4] = {
    {1, 0, 0, 0}, 
    {0, 1, -1, 1}, 
    {-1, 1, 1, 0}, 
    {0, 0, 0, -1}
};
const float B_T[4][4] = {
    {1, 0, -1, 0}, 
    {0, 1, 1, 0}, 
    {0, -1, 1, 0}, 
    {0, 1, 0, -1}
};
const float A[4][2] = {
    {1, 0}, 
    {1, 1}, 
    {1, -1}, 
    {0, -1}
};
const float A_T[2][4] = {
    {1, 1, 1, 0}, 
    {0, 1, -1, -1}
};
/*
利用手写展开和向量化用来计算 V 矩阵
*/
static inline void compute_V_neon(const float* d, float* v){
    // 加载输入
    const float32x4_t d0 = vld1q_f32(d + 0);  // d[0]..d[3]
    const float32x4_t d1 = vld1q_f32(d + 4);  // d[4]..d[7]
    const float32x4_t d2 = vld1q_f32(d + 8);  // d[8]..d[11]
    const float32x4_t d3 = vld1q_f32(d + 12); // d[12]..d[15]

    // 计算第一个变换
    float32x4_t tmp0 = vsubq_f32(d0, d2); // d[0] - d[8], d[1] - d[9], ...
    float32x4_t tmp1 = vaddq_f32(d1, d2); // d[1] + d[5], d[9] + d[13], ...
    float32x4_t tmp2 = vsubq_f32(d2, d1); // d[1] - d[5], d[9] - d[13], ...
    float32x4_t tmp3 = vsubq_f32(d1, d3); // d[0] - d[1], d[8] - d[9], ...
   
    // 转置操作，将结果变换为列优先
    float32x4x2_t P0 = vtrnq_f32(tmp0,tmp1);
    float32x4x2_t P1 = vtrnq_f32(tmp2,tmp3);

    // 重组四个列向量
    float32x4_t t_col0 = vcombine_f32(vget_low_f32(P0.val[0]), vget_low_f32(P1.val[0]));  // {t00,t10,t20,t30}
    float32x4_t t_col1 = vcombine_f32(vget_low_f32(P0.val[1]), vget_low_f32(P1.val[1]));  // {t01,t11,t21,t31}
    float32x4_t t_col2 = vcombine_f32(vget_high_f32(P0.val[0]), vget_high_f32(P1.val[0])); // {t02,t12,t22,t32}
    float32x4_t t_col3 = vcombine_f32(vget_high_f32(P0.val[1]), vget_high_f32(P1.val[1])); // {t03,t13,t23,t33}
    // 计算第二个变换
    float32x4_t v_col0 = vsubq_f32(t_col0, t_col2);
    float32x4_t v_col1 = vaddq_f32(t_col1, t_col2);
    float32x4_t v_col2 = vsubq_f32(t_col2, t_col1);
    float32x4_t v_col3 = vsubq_f32(t_col1, t_col3);
    // 转置为行优先
    float32x4x2_t Q0 = vtrnq_f32(v_col0, v_col1);
    float32x4x2_t Q1 = vtrnq_f32(v_col2, v_col3);
    // 重组四个行向量
    float32x4_t v_row0 = vcombine_f32(vget_low_f32(Q0.val[0]), vget_low_f32(Q1.val[0]));
    float32x4_t v_row1 = vcombine_f32(vget_low_f32(Q0.val[1]), vget_low_f32(Q1.val[1]));
    float32x4_t v_row2 = vcombine_f32(vget_high_f32(Q0.val[0]), vget_high_f32(Q1.val[0]));
    float32x4_t v_row3 = vcombine_f32(vget_high_f32(Q0.val[1]), vget_high_f32(Q1.val[1]));
    // 将结果存储到输出矩阵 v
    vst1q_f32(v + 0, v_row0);
    vst1q_f32(v + 4, v_row1);
    vst1q_f32(v + 8, v_row2);
    vst1q_f32(v + 12, v_row3);
}

static inline void compute_U_neon(const float* g, float* u) {
    // 实际上我们只用存储的前三个元素
    const float32x4_t g0 = vld1q_f32(g + 0);
    const float32x4_t g1 = vld1q_f32(g + 3); // g+3 指向第 4 个元素
    const float32x4_t g2 = vld1q_f32(g + 6); // g+6 指向第 7 个元素

    const float32x4_t v_half = vdupq_n_f32(0.5f); // 常量0.5

    // 计算第一个变换
    const float32x4_t tmp_u0 = g0;
    const float32x4_t tmp_u1 = vmulq_f32(vaddq_f32(vaddq_f32(g0, g1), g2), v_half);
    const float32x4_t tmp_u2 = vmulq_f32(vaddq_f32(vsubq_f32(g0, g1), g2), v_half);
    const float32x4_t tmp_u3 = g2;

    // 转置操作，将结果变换为列优先
    float32x4x2_t P0 = vtrnq_f32(tmp_u0, tmp_u1);
    float32x4x2_t P1 = vtrnq_f32(tmp_u2, tmp_u3);

    // 重组三个列向量
    float32x4_t t_col0 = vcombine_f32(vget_low_f32(P0.val[0]), vget_low_f32(P1.val[0]));
    float32x4_t t_col1 = vcombine_f32(vget_low_f32(P0.val[1]), vget_low_f32(P1.val[1]));
    float32x4_t t_col2 = vcombine_f32(vget_high_f32(P0.val[0]), vget_high_f32(P1.val[0]));
    
    // 计算第二个变换
    const float32x4_t u_col0 = t_col0;
    const float32x4_t u_col1 = vmulq_f32(vaddq_f32(vaddq_f32(t_col0, t_col1), t_col2), v_half);
    const float32x4_t u_col2 = vmulq_f32(vaddq_f32(vsubq_f32(t_col0, t_col1), t_col2), v_half);
    const float32x4_t u_col3 = t_col2;

    // 转置为行优先
    float32x4x2_t Q0 = vtrnq_f32(u_col0, u_col1);
    float32x4x2_t Q1 = vtrnq_f32(u_col2, u_col3);
    
    // 重组四个行向量
    float32x4_t u_row0 = vcombine_f32(vget_low_f32(Q0.val[0]), vget_low_f32(Q1.val[0]));
    float32x4_t u_row1 = vcombine_f32(vget_low_f32(Q0.val[1]), vget_low_f32(Q1.val[1]));
    float32x4_t u_row2 = vcombine_f32(vget_high_f32(Q0.val[0]), vget_high_f32(Q1.val[0]));
    float32x4_t u_row3 = vcombine_f32(vget_high_f32(Q0.val[1]), vget_high_f32(Q1.val[1]));
    
    // 将结果存储到输出矩阵 u
    vst1q_f32(u + 0, u_row0);
    vst1q_f32(u + 4, u_row1);
    vst1q_f32(u + 8, u_row2);
    vst1q_f32(u + 12, u_row3);
}
static inline void compute_output_neon(const float* mm, float* out) {
    // 加载输入
    const float32x4_t m0 = vld1q_f32_x4(mm).val[0];
    const float32x4_t m1 = vld1q_f32_x4(mm).val[1];
    const float32x4_t m2 = vld1q_f32_x4(mm).val[2];
    const float32x4_t m3 = vld1q_f32_x4(mm).val[3];

    // 计算变换结果
    const float32x4_t tmp_m0_vec = vaddq_f32(vaddq_f32(m0, m1), m2);
    const float32x4_t tmp_m1_vec = vsubq_f32(vsubq_f32(m1, m2), m3);

    // 提取结果
    float tm00 = vgetq_lane_f32(tmp_m0_vec, 0);
    float tm01 = vgetq_lane_f32(tmp_m0_vec, 1);
    float tm02 = vgetq_lane_f32(tmp_m0_vec, 2);
    float tm03 = vgetq_lane_f32(tmp_m0_vec, 3);

    float tm10 = vgetq_lane_f32(tmp_m1_vec, 0);
    float tm11 = vgetq_lane_f32(tmp_m1_vec, 1);
    float tm12 = vgetq_lane_f32(tmp_m1_vec, 2);
    float tm13 = vgetq_lane_f32(tmp_m1_vec, 3);
    
    // 存储最终输出
    out[0] = tm00 + tm01 + tm02;
    out[1] = tm01 - tm02 - tm03;
    out[2] = tm10 + tm11 + tm12;
    out[3] = tm11 - tm12 - tm13;
}


void sgemm(const float* A, const float* B, float* out, 
           const int M, const int K, const int N) {

    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k) {
                sum += A[i * K + k] * B[k * N + j];  
            }
            out[i * N + j] = sum;
        }
    }
}

void sgemm_parallel(const float* A, const float* B, float* out,
                    const int M, const int K, const int N) {
    for (int i = 0; i < M * N; ++i) {
        out[i] = 0.0f;
    }

    #pragma omp parallel for collapse(2)
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            for (int k = 0; k < K; ++k) {
                out[i * N + j] += A[i * K + k] * B[k * N + j];
            }
        }
    }
}

/** 
 * @brief Winograd Implementation of F(2x2, 3x3)
 * @param image: [batch * C * inHeight * inWidth]
 * @param filter: [K * C * 3 * 3]
 * @param out: Output result. Shape is [batch * K * outHeight * outWidth]
 * @param U: [4 * 4 * K * C], intermediate transformed filters
 * @param V: [4 * 4 * C * P], intermediate transformed image
 * @param M: [4 * 4 * K * P], intermediate result
 * @param inHeight: Height of input image
 * @param inWidth: Width of input image
 * @param C: Number of channels in input image
 * @param K: Number of filters
 * @param N: Batch size
 */
void winograd_conv(const float* restrict image, const float* restrict filter, float* restrict out,
                   float* restrict U, float* restrict V, float* restrict M,
                   const int inHeight, const int inWidth, const int C, const int K, const int N) {
    const int outHeight = inHeight - 2; // output height
    const int outWidth = inWidth - 2; // output width
    const int sizeI = inHeight * inWidth; // size of input image
    const int sizeF = 3 * 3; // size of filter
    const int sizeO = outHeight * outWidth; // size of output
    const int P = outHeight / 2 * outWidth / 2 * N; // size of output in blocks

    float tmp_u[12]; // 4 * 3
    float u[16];     // 4 * 4;
    
    // Transform filters and scatter to U
    // U[:, :, k, c] = G * filters[k, c, :, :] * G^T
    #pragma omp parallel for private(tmp_u, u)
    for (int k = 0; k < K; ++k) {
        for (int c = 0; c < C; ++c) {
            const float* filters_ptr = filter + (k * C + c) * sizeF;

            compute_U_neon(filters_ptr, u);
            
            for (int xi = 0; xi < 4; ++xi) {
                for (int nu = 0; nu < 4; ++nu) {
                    U[((xi * 4 + nu) * K + k) * C + c] = u[xi * 4 + nu];

                }
            }
        }
    }


    // Transform image and scatter to V
    // V[:, :, c, p] = B^T * image[c, b, :, :] * B
    float tmp_v[16];
    float d[16]; // d: [4 * 4];
    float v[16]; // v: [4 * 4];
    #pragma omp parallel for collapse(2) private(tmp_v, d, v)
    for (int n = 0; n < N; ++n) {
        for (int c = 0; c < C; ++c) {
           for(int yy = 0; yy < outHeight / 2; yy += BH){
                for(int xx = 0; xx < outWidth / 2; xx += BW){
                    for(int y = yy;y < yy + BH && y < outHeight /2; ++y){
                        for(int x = xx; x < xx + BW && x < outWidth / 2; ++x){
                            // Generate d_cb
                            for (int iy = 0; iy < 4; ++iy) {
                                for (int ix = 0; ix < 4; ++ix) {
                                    d[iy * 4 + ix] = image[(n * C + c) * sizeI +
                                                            (y * 2 + iy) * inWidth + (x * 2 + ix)];
                                }
                            }
                            // Compute V
                            compute_V_neon(d, v);
                            
                            int b = ((n * outHeight / 2) + y) * outWidth / 2 + x;
                            for (int xi = 0; xi < 4; ++xi) {
                                for (int nu = 0; nu < 4; ++nu) {
                                    V[((xi * 4 + nu) * C + c) * P + b] = v[xi * 4 + nu];
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    // M[xi, nu, :, :] = U[xi, nu, :, :] * V[xi, nu, :, :]
    #pragma omp parallel for collapse(2)
    for (int xi = 0; xi < 4; ++xi) {
        for (int nu = 0; nu < 4; ++nu) {
            float* M_ptr = M + (xi * 4 + nu) * K * P;
            float* U_ptr = U + (xi * 4 + nu) * K * C;
            float* V_ptr = V + (xi * 4 + nu) * C * P;
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                K, P, C, 1.0f, U_ptr, C, V_ptr, P, 0.0f, M_ptr, P);
        }
    }

    // Gather output and apply inverse transformation
    // Y = A^T * m * A
    float mm[16];      // 4 * 4
    float tmp_m[8];    // 2 * 4
    float temp_out[4]; // 2 * 2
    #pragma omp parallel for collapse(2) private(mm, temp_out, tmp_m)
    for(int n = 0;n < N ;++n){
        for(int k = 0; k < K; ++k){
            for(int yy = 0; yy < outHeight / 2; yy += BH){
                for(int xx = 0; xx < outWidth / 2; xx += BW){

                    for(int y = yy; y < yy + BH && y < outHeight / 2; ++y){
                        for(int x = xx; x < xx + BW && x < outWidth / 2; ++x){
                            int b = (n * outHeight / 2 + y) * outWidth / 2 + x;
                            for(int xi = 0; xi < 4; ++xi){
                                for(int nu = 0; nu < 4; ++nu){
                                    mm[xi * 4 + nu] = M[((xi * 4 + nu) * K + k) * P + b];
                                }
                            }
                           
                            compute_output_neon(mm, temp_out);

                            // --- SCATTER ---
                            out[((n * K + k) * outHeight + y * 2) * outWidth + x * 2] = temp_out[0];
                            out[((n * K + k) * outHeight + y * 2) * outWidth + x * 2 + 1] = temp_out[1];
                            out[((n * K + k) * outHeight + y * 2 + 1) * outWidth + x * 2] = temp_out[2];
                            out[((n * K + k) * outHeight + y * 2 + 1) * outWidth + x * 2 + 1] = temp_out[3];
                        }
                    }
                }
            }
        }
    }

    /*
    无效的优化结果
    */
    // #pragma omp parallel for collapse(2)
    // for(int n = 0;n < N; ++n){
    //     for(int  yy = 0;yy < outHeight / 2; yy += BH){
    //         for(int xx = 0; xx < outWidth / 2; xx += BW){
    //             for(int kk = 0; kk < K; kk += BK){
    //                 // 临时变量，对每个 (n,y,x) tile 任务是私有的

    //                 // 对每个 (n, y, x, k) tile 任务
    //                 for (int y = yy; y < yy + BH && y < outHeight / 2; ++y){
    //                     for (int x = xx; x < xx + BW && x < outWidth / 2; ++x) {
    //                         int b = (n * outHeight / 2 + y) * outWidth / 2 + x;
                            
    //                         for (int k = kk; k < kk + BK && k < K; ++k) {

                                
    //                             // Gather M for current (n, y, x, k)
    //                             for (int xi = 0; xi < 4; ++xi) {
    //                                 for (int nu = 0; nu < 4; ++nu) {
    //                                     mm[xi * 4 + nu] = M[((xi * 4 + nu) * K + k) * P + b];
    //                                 }
    //                             }
                                
    //                             float tmp_m00 = mm[0] + mm[4] + mm[8];
    //                             float tmp_m01 = mm[1] + mm[5] + mm[9];
    //                             float tmp_m02 = mm[2] + mm[6] + mm[10];
    //                             float tmp_m03 = mm[3] + mm[7] + mm[11];
                                
    //                             float tmp_m10 = mm[4] - mm[8] - mm[12];
    //                             float tmp_m11 = mm[5] - mm[9] - mm[13];
    //                             float tmp_m12 = mm[6] - mm[10] - mm[14];
    //                             float tmp_m13 = mm[7] - mm[11] - mm[15];
                                
    //                             float out00 = tmp_m00 + tmp_m01 + tmp_m02;
    //                             float out01 = tmp_m01 - tmp_m02 - tmp_m03;
    //                             float out10 = tmp_m10 + tmp_m11 + tmp_m12;
    //                             float out11 = tmp_m11 - tmp_m12 - tmp_m13;

    //                             // --- SCATTER ---
    //                             out[((n * K + k) * outHeight + y * 2) * outWidth + x * 2] = out00;
    //                             out[((n * K + k) * outHeight + y * 2) * outWidth + x * 2 + 1] = out01;
    //                             out[((n * K + k) * outHeight + y * 2 + 1) * outWidth + x * 2] = out10;
    //                             out[((n * K + k) * outHeight + y * 2 + 1) * outWidth + x * 2 + 1] = out11;
    //                         }
    //                     }
    //                 }
    //             }
    //         }
    //     }
    // }

}
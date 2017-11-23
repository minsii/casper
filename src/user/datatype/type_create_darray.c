/* -*- Mode: C; c-basic-offset:4 ; -*- */
/*
 * (C) 2017 by Argonne National Laboratory.
 *     See COPYRIGHT in top-level directory.
 */


#include <stdio.h>
#include <stdlib.h>
#include "cspu.h"

static inline int type_darray_impl(int size, int rank, int ndims,
                                   const int array_of_gsizes[],
                                   const int array_of_distribs[],
                                   const int array_of_dargs[],
                                   const int array_of_psizes[],
                                   int order, MPI_Datatype oldtype, MPI_Datatype * newtype)
{
    int mpi_errno = MPI_SUCCESS;
    CSP_cwp_pkt_t pkt;
    CSP_cwp_dtype_darray_regist_pkt_t *regist_pkt = &pkt.u.fnc_dtype_darray_regist;
    int i = 0, lrank = 0, off = 0;
    char *packed_params = NULL;
    int packed_param_sz = 0;
    int *arr_gsizes_ptr = NULL, *arr_distribs_ptr = NULL;
    int *arr_dargs_ptr = NULL, *arr_psizes_ptr = NULL;
    MPI_Datatype *g_oldtype_ptr = NULL;

    /* Locally create new datatype */
    CSP_CALLMPI(JUMP, PMPI_Type_create_darray(size, rank, ndims, array_of_gsizes, array_of_distribs,
                                              array_of_dargs, array_of_psizes, order, oldtype,
                                              newtype));

    /* Send command and static parameters to root ghost.
     * Do not mlock ghost because only one node. */
    CSP_CALLMPI(JUMP, PMPI_Comm_rank(CSP_PROC.local_comm, &lrank));

    CSP_cwp_init_pkt(CSP_CWP_FNC_DTYPE_DARRAY_REGIST, &pkt);
    regist_pkt->user_local_root = lrank;
    regist_pkt->param.size = size;
    regist_pkt->param.rank = rank;
    regist_pkt->param.ndims = ndims;
    regist_pkt->param.order = order;

    mpi_errno = CSPU_cwp_issue(&pkt);
    CSP_CHKMPIFAIL_JUMP(mpi_errno);

    CSP_DBG_PRINT("DTYPE: create darray 0x%lx (ndims %d, order 0x%x, oldtype 0x%lx)\n",
                  (MPI_Aint) newtype, regist_pkt->param.ndims, regist_pkt->param.order,
                  (MPI_Aint) oldtype);

    /* Send extra parameter in one packed buffer. */
    packed_param_sz = regist_pkt->param.ndims * 4 * sizeof(int) + sizeof(MPI_Datatype);
    packed_params = (char *) CSP_calloc(packed_param_sz, sizeof(char));

    arr_gsizes_ptr = (int *) &packed_params[off];
    memcpy(arr_gsizes_ptr, array_of_gsizes, sizeof(int) * ndims);
    off += sizeof(int) * ndims;

    arr_distribs_ptr = (int *) &packed_params[off];
    memcpy(arr_distribs_ptr, array_of_distribs, sizeof(int) * ndims);
    off += sizeof(int) * ndims;

    arr_dargs_ptr = (int *) &packed_params[off];
    memcpy(arr_dargs_ptr, array_of_dargs, sizeof(int) * ndims);
    off += sizeof(int) * ndims;

    arr_psizes_ptr = (int *) &packed_params[off];
    memcpy(arr_psizes_ptr, array_of_psizes, sizeof(int) * ndims);
    off += sizeof(int) * ndims;

    g_oldtype_ptr = (MPI_Datatype *) & packed_params[off];

    for (i = 0; i < CSP_ENV.num_g; i++) {
        mpi_errno = CSPU_datatype_get_g_handle(oldtype, i, g_oldtype_ptr);
        CSP_CHKMPIFAIL_JUMP(mpi_errno);

        /* Consider only a few ghosts exist, use blocking send here to reuse buffer. */
        CSP_CALLMPI(JUMP, PMPI_Send(packed_params, packed_param_sz, MPI_BYTE,
                                    CSP_PROC.user.g_lranks[i], CSP_CWP_PARAM_TAG,
                                    CSP_PROC.local_comm));
    }

    mpi_errno = CSPU_datatype_regist_complete(*newtype);
    CSP_CHKMPIFAIL_JUMP(mpi_errno);

  fn_exit:
    if (packed_params)
        free(packed_params);
    return mpi_errno;
  fn_fail:
    goto fn_exit;
}

int MPI_Type_create_darray(int size, int rank, int ndims,
                           const int array_of_gsizes[],
                           const int array_of_distribs[],
                           const int array_of_dargs[],
                           const int array_of_psizes[],
                           int order, MPI_Datatype oldtype, MPI_Datatype * newtype)
{
    int mpi_errno = MPI_SUCCESS;

    /* Skip internal processing when disabled */
    if (CSP_IS_DISABLED || CSP_IS_MODE_DISABLED(PT2PT))
        return PMPI_Type_create_darray(size, rank, ndims, array_of_gsizes, array_of_distribs,
                                       array_of_dargs, array_of_psizes, order, oldtype, newtype);

    mpi_errno = type_darray_impl(size, rank, ndims, array_of_gsizes, array_of_distribs,
                                 array_of_dargs, array_of_psizes, order, oldtype, newtype);
    CSP_CHKMPIFAIL_JUMP(mpi_errno);

  fn_exit:
    return mpi_errno;
  fn_fail:
    goto fn_exit;
}

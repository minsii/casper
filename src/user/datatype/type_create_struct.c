/* -*- Mode: C; c-basic-offset:4 ; -*- */
/*
 * (C) 2017 by Argonne National Laboratory.
 *     See COPYRIGHT in top-level directory.
 */


#include <stdio.h>
#include <stdlib.h>
#include "cspu.h"

static inline int type_struct_impl(int count,
                                   const int array_of_blocklengths[],
                                   const MPI_Aint array_of_displacements[],
                                   const MPI_Datatype array_of_types[], MPI_Datatype * newtype)
{
    int mpi_errno = MPI_SUCCESS;
    CSP_cwp_pkt_t pkt;
    CSP_cwp_dtype_struct_regist_pkt_t *regist_pkt = &pkt.u.fnc_dtype_struct_regist;
    int i = 0, lrank = 0, off = 0;
    char *packed_params = NULL;
    int packed_param_sz = 0;
    int *arr_blens_ptr = NULL;
    MPI_Aint *arr_disps_ptr = NULL;
    MPI_Datatype *arr_types_ptr = NULL;

    /* Locally create new datatype */
    CSP_CALLMPI(JUMP, PMPI_Type_create_struct(count, array_of_blocklengths, array_of_displacements,
                                              array_of_types, newtype));

    /* Send command and static parameters to root ghost.
     * Do not mlock ghost because only one node. */
    CSP_CALLMPI(JUMP, PMPI_Comm_rank(CSP_PROC.local_comm, &lrank));

    CSP_cwp_init_pkt(CSP_CWP_FNC_DTYPE_STRUCT_REGIST, &pkt);
    regist_pkt->user_local_root = lrank;
    regist_pkt->param.count = count;

    mpi_errno = CSPU_cwp_issue(&pkt);
    CSP_CHKMPIFAIL_JUMP(mpi_errno);

    CSP_DBG_PRINT("DTYPE: create struct 0x%lx (count %d)\n", (MPI_Aint) newtype,
                  regist_pkt->param.count);

    /* Send extra parameter in one packed buffer. */
    packed_param_sz = regist_pkt->param.count * sizeof(int) +
        regist_pkt->param.count * sizeof(MPI_Aint) + regist_pkt->param.count * sizeof(MPI_Datatype);
    packed_params = (char *) CSP_calloc(packed_param_sz, sizeof(char));

    arr_blens_ptr = (int *) &packed_params[off];
    memcpy(arr_blens_ptr, array_of_blocklengths, sizeof(int) * count);
    off += sizeof(int) * count;

    arr_disps_ptr = (MPI_Aint *) & packed_params[off];
    memcpy(arr_disps_ptr, array_of_displacements, sizeof(MPI_Aint) * count);
    off += sizeof(MPI_Aint) * count;

    arr_types_ptr = (MPI_Datatype *) & packed_params[off];
    memcpy(arr_types_ptr, array_of_types, sizeof(MPI_Datatype) * count);

    for (i = 0; i < CSP_ENV.num_g; i++) {
        int c = 0;
        for (c = 0; c < count; c++) {
            mpi_errno = CSPU_datatype_get_g_handle(array_of_types[c], i, &arr_types_ptr[c]);
            CSP_CHKMPIFAIL_JUMP(mpi_errno);
        }

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


int MPI_Type_create_struct(int count,
                           const int array_of_blocklengths[],
                           const MPI_Aint array_of_displacements[],
                           const MPI_Datatype array_of_types[], MPI_Datatype * newtype)
{
    int mpi_errno = MPI_SUCCESS;

    /* Skip internal processing when disabled */
    if (CSP_IS_DISABLED || CSP_IS_MODE_DISABLED(PT2PT))
        return PMPI_Type_create_struct(count, array_of_blocklengths, array_of_displacements,
                                       array_of_types, newtype);

    mpi_errno = type_struct_impl(count, array_of_blocklengths, array_of_displacements,
                                 array_of_types, newtype);
    CSP_CHKMPIFAIL_JUMP(mpi_errno);

  fn_exit:
    return mpi_errno;
  fn_fail:
    goto fn_exit;
}

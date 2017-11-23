/* -*- Mode: C; c-basic-offset:4 ; -*- */
/*
 * (C) 2017 by Argonne National Laboratory.
 *     See COPYRIGHT in top-level directory.
 */

#include <stdio.h>
#include <stdlib.h>
#include "cspu.h"

static inline int type_indexed_impl(int count, const int *array_of_blocklengths,
                                    const int *array_of_displacements,
                                    MPI_Datatype oldtype, MPI_Datatype * newtype)
{
    int mpi_errno = MPI_SUCCESS;
    CSP_cwp_pkt_t pkt;
    CSP_cwp_dtype_indexed_regist_pkt_t *regist_pkt = &pkt.u.fnc_dtype_indexed_regist;
    int i = 0, lrank = 0, off = 0;
    char *packed_params = NULL;
    int packed_param_sz = 0;
    int *arr_blens_ptr = NULL, *arr_disps_ptr = NULL;
    MPI_Datatype *g_oldtype_ptr = NULL;

    /* Locally create new datatype */
    CSP_CALLMPI(JUMP, PMPI_Type_indexed(count, array_of_blocklengths, array_of_displacements,
                                        oldtype, newtype));

    /* Send command and static parameters to root ghost.
     * Do not mlock ghost because only one node. */
    CSP_CALLMPI(JUMP, PMPI_Comm_rank(CSP_PROC.local_comm, &lrank));

    CSP_cwp_init_pkt(CSP_CWP_FNC_DTYPE_INDEXED_REGIST, &pkt);
    regist_pkt->user_local_root = lrank;
    regist_pkt->param.count = count;

    mpi_errno = CSPU_cwp_issue(&pkt);
    CSP_CHKMPIFAIL_JUMP(mpi_errno);

    CSP_DBG_PRINT("DTYPE: create indexed 0x%lx (count %d, oldtype 0x%lx)\n",
                  (MPI_Aint) newtype, regist_pkt->param.count, (MPI_Aint) oldtype);

    /* Prepare other parameters and send to each ghost. */
    packed_param_sz = count * sizeof(int) * 2 + sizeof(MPI_Datatype);
    packed_params = (char *) CSP_calloc(packed_param_sz, sizeof(char));

    arr_blens_ptr = (int *) &packed_params[off];
    memcpy(arr_blens_ptr, array_of_blocklengths, sizeof(int) * count);
    off += sizeof(int) * count;

    arr_disps_ptr = (int *) &packed_params[off];
    memcpy(arr_disps_ptr, array_of_displacements, sizeof(int) * count);
    off += sizeof(int) * count;

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

int MPI_Type_indexed(int count, const int *array_of_blocklengths, const int *array_of_displacements,
                     MPI_Datatype oldtype, MPI_Datatype * newtype)
{
    int mpi_errno = MPI_SUCCESS;

    /* Skip internal processing when disabled */
    if (CSP_IS_DISABLED || CSP_IS_MODE_DISABLED(PT2PT))
        return PMPI_Type_indexed(count, array_of_blocklengths, array_of_displacements,
                                 oldtype, newtype);

    mpi_errno = type_indexed_impl(count, array_of_blocklengths, array_of_displacements,
                                  oldtype, newtype);
    CSP_CHKMPIFAIL_JUMP(mpi_errno);

  fn_exit:
    return mpi_errno;
  fn_fail:
    goto fn_exit;
}

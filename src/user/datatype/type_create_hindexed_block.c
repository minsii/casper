/* -*- Mode: C; c-basic-offset:4 ; -*- */
/*
 * (C) 2017 by Argonne National Laboratory.
 *     See COPYRIGHT in top-level directory.
 */


#include <stdio.h>
#include <stdlib.h>
#include "cspu.h"

static inline int type_hindexed_block_impl(int count, int blocklength,
                                           const MPI_Aint array_of_displacements[],
                                           MPI_Datatype oldtype, MPI_Datatype * newtype)
{
    int mpi_errno = MPI_SUCCESS;
    CSP_cwp_pkt_t pkt;
    CSP_cwp_dtype_hidx_blk_regist_pkt_t *regist_pkt = &pkt.u.fnc_dtype_hidx_blk_regist;
    int i = 0, lrank = 0, off = 0;
    char *packed_params = NULL;
    int packed_param_sz = 0;
    MPI_Aint *arr_disps_ptr = NULL;
    MPI_Datatype *g_oldtype_ptr = NULL;

    /* Locally create new datatype */
    CSP_CALLMPI(JUMP, PMPI_Type_create_hindexed_block(count, blocklength,
                                                      array_of_displacements, oldtype, newtype));

    /* Send command and static parameters to root ghost.
     * Do not mlock ghost because only one node. */
    CSP_CALLMPI(JUMP, PMPI_Comm_rank(CSP_PROC.local_comm, &lrank));

    CSP_cwp_init_pkt(CSP_CWP_FNC_DTYPE_HIDX_BLK_REGIST, &pkt);
    regist_pkt->user_local_root = lrank;
    regist_pkt->param.count = count;
    regist_pkt->param.blocklength = blocklength;

    mpi_errno = CSPU_cwp_issue(&pkt);
    CSP_CHKMPIFAIL_JUMP(mpi_errno);

    CSP_DBG_PRINT("DTYPE: create hindexed_block 0x%lx (count %d, blen %d, oldtype 0x%lx)\n",
                  (MPI_Aint) newtype, regist_pkt->param.count, regist_pkt->param.blocklength,
                  (MPI_Aint) oldtype);

    /* Prepare other parameters and send to each ghost. */
    packed_param_sz = count * sizeof(MPI_Aint) + sizeof(MPI_Datatype);
    packed_params = (char *) CSP_calloc(packed_param_sz, sizeof(char));

    arr_disps_ptr = (MPI_Aint *) & packed_params[off];
    memcpy(arr_disps_ptr, array_of_displacements, sizeof(MPI_Aint) * count);
    off += sizeof(MPI_Aint) * count;

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

int MPI_Type_create_hindexed_block(int count, int blocklength,
                                   const MPI_Aint array_of_displacements[],
                                   MPI_Datatype oldtype, MPI_Datatype * newtype)
{
    int mpi_errno = MPI_SUCCESS;

    /* Skip internal processing when disabled */
    if (CSP_IS_DISABLED || CSP_IS_MODE_DISABLED(PT2PT))
        return PMPI_Type_create_hindexed_block(count, blocklength, array_of_displacements,
                                               oldtype, newtype);

    mpi_errno = type_hindexed_block_impl(count, blocklength, array_of_displacements,
                                         oldtype, newtype);
    CSP_CHKMPIFAIL_JUMP(mpi_errno);

  fn_exit:
    return mpi_errno;
  fn_fail:
    goto fn_exit;
}

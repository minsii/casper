/* -*- Mode: C; c-basic-offset:4 ; -*- */
/*
 * (C) 2017 by Argonne National Laboratory.
 *     See COPYRIGHT in top-level directory.
 */

#include <stdio.h>
#include <stdlib.h>
#include "cspg.h"

static inline int dtype_hidx_blk_regist_impl(CSP_cwp_dtype_hidx_blk_regist_pkt_t * regist_pkt)
{
    int mpi_errno = MPI_SUCCESS;
    MPI_Datatype newtype = MPI_DATATYPE_NULL;
    char *packed_params = NULL;
    int packed_param_sz = 0;
    MPI_Aint *arr_disps_ptr = NULL;
    MPI_Datatype *oldtype_ptr = NULL;
    int off = 0;

    /* Receive extra parameter in one packed buffer. */
    packed_param_sz = regist_pkt->param.count * sizeof(MPI_Aint) + sizeof(MPI_Datatype);
    packed_params = (char *) CSP_calloc(packed_param_sz, sizeof(char));

    CSP_CALLMPI(JUMP, PMPI_Recv(packed_params, packed_param_sz, MPI_CHAR,
                                regist_pkt->user_local_root, CSP_CWP_PARAM_TAG,
                                CSP_PROC.local_comm, MPI_STATUS_IGNORE));

    arr_disps_ptr = (MPI_Aint *) & packed_params[off];
    off += sizeof(MPI_Aint) * regist_pkt->param.count;

    oldtype_ptr = (MPI_Datatype *) & packed_params[off];

    /* Create new datatype */
    CSP_CALLMPI(JUMP, PMPI_Type_create_hindexed_block(regist_pkt->param.count,
                                                      regist_pkt->param.blocklength,
                                                      arr_disps_ptr, *oldtype_ptr, &newtype));
    CSP_ASSERT(newtype != MPI_DATATYPE_NULL);
    CSPG_DBG_PRINT("DTYPE: registered hindexed_block 0x%lx (count %d, blen %d, oldtype 0x%lx)\n",
                   (MPI_Aint) newtype, regist_pkt->param.count, regist_pkt->param.blocklength,
                   (MPI_Aint) * oldtype_ptr);

    /* Send back the new datatype handle. */
    CSP_CALLMPI(JUMP, PMPI_Send(&newtype, sizeof(MPI_Datatype), MPI_BYTE,
                                regist_pkt->user_local_root, CSP_CWP_PARAM_TAG,
                                CSP_PROC.local_comm));
  fn_exit:
    return mpi_errno;
  fn_fail:
    goto fn_exit;
}

/* *INDENT-OFF* */
CSPG_DEFINE_GENERIC_CWP_ROOT_HANDLER(dtype_hidx_blk_regist)
CSPG_DEFINE_GENERIC_CWP_HANDLER(dtype_hidx_blk_regist)
/* *INDENT-ON* */

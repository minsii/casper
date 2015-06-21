/* -*- Mode: C; c-basic-offset:4 ; -*- */
/*
 * (C) 2014 by Argonne National Laboratory.
 *     See COPYRIGHT in top-level directory.
 */

#include <stdio.h>
#include <stdlib.h>
#include "csp.h"

#ifdef CSP_ENABLE_LOCAL_LOCK_OPT
static inline int CSP_win_unlock_self_impl(CSP_win * ug_win)
{
    int mpi_errno = MPI_SUCCESS;

#ifdef CSP_ENABLE_SYNC_ALL_OPT
    /* unlockall already released window for local target */
#else
    int user_rank;
    PMPI_Comm_rank(ug_win->user_comm, &user_rank);

    if (ug_win->is_self_locked) {
        /* We need also release the lock of local rank */

        CSP_DBG_PRINT("[%d]unlock self(%d, local win 0x%x)\n", user_rank,
                      ug_win->my_rank_in_ug_comm, ug_win->my_ug_win);
        mpi_errno = PMPI_Win_unlock(ug_win->my_rank_in_ug_comm, ug_win->my_ug_win);
        if (mpi_errno != MPI_SUCCESS)
            return mpi_errno;
    }
#endif

    ug_win->is_self_locked = 0;
    return mpi_errno;
}
#endif

int MPI_Win_unlock(int target_rank, MPI_Win win)
{
    CSP_win *ug_win;
    int mpi_errno = MPI_SUCCESS;
    int user_rank;
    int k;

    CSP_DBG_PRINT_FCNAME();

    CSP_fetch_ug_win_from_cache(win, ug_win);

    if (ug_win == NULL) {
        /* normal window */
        return PMPI_Win_unlock(target_rank, win);
    }

    /* casper window starts */

    if (target_rank == MPI_PROC_NULL)
        goto fn_exit;

    CSP_assert((ug_win->info_args.epoch_type & CSP_EPOCH_LOCK) ||
               (ug_win->info_args.epoch_type & CSP_EPOCH_LOCK_ALL));

    PMPI_Comm_rank(ug_win->user_comm, &user_rank);

    ug_win->targets[target_rank].remote_lock_assert = 0;

    /* Unlock all ghost processes in every ug-window of target process. */
#ifdef CSP_ENABLE_SYNC_ALL_OPT

    /* Optimization for MPI implementations that have optimized lock_all.
     * However, user should be noted that, if MPI implementation issues lock messages
     * for every target even if it does not have any operation, this optimization
     * could lose performance and even lose asynchronous! */

    CSP_DBG_PRINT("[%d]unlock_all(ug_win 0x%x), instead of target rank %d\n",
                  user_rank, ug_win->targets[target_rank].ug_win, target_rank);
    mpi_errno = PMPI_Win_unlock_all(ug_win->targets[target_rank].ug_win);
    if (mpi_errno != MPI_SUCCESS)
        goto fn_fail;
#else
    for (k = 0; k < CSP_ENV.num_g; k++) {
        int target_g_rank_in_ug = ug_win->targets[target_rank].g_ranks_in_ug[k];

        CSP_DBG_PRINT("[%d]unlock(Ghost(%d), ug_win 0x%x), instead of "
                      "target rank %d\n", user_rank, target_g_rank_in_ug,
                      ug_win->targets[target_rank].ug_win, target_rank);

        mpi_errno = PMPI_Win_unlock(target_g_rank_in_ug, ug_win->targets[target_rank].ug_win);
        if (mpi_errno != MPI_SUCCESS)
            goto fn_fail;
    }
#endif


#ifdef CSP_ENABLE_LOCAL_LOCK_OPT
    /* If target is itself, we need also release the lock of local rank  */
    if (user_rank == target_rank && ug_win->is_self_locked) {
        mpi_errno = CSP_win_unlock_self_impl(ug_win);
        if (mpi_errno != MPI_SUCCESS)
            goto fn_fail;
    }
#endif

#if defined(CSP_ENABLE_RUNTIME_LOAD_OPT)
    int j;
    for (j = 0; j < ug_win->targets[target_rank].num_segs; j++) {
        ug_win->targets[target_rank].segs[j].main_lock_stat = CSP_MAIN_LOCK_RESET;
    }
#endif

    /* Decrease lock/lockall counter, change epoch status only when counter
     * become 0. */
    ug_win->lock_counter--;
    if (ug_win->lockall_counter == 0 && ug_win->lock_counter == 0) {
        CSP_DBG_PRINT("all locks are cleared ! no epoch now\n");
        ug_win->epoch_stat = CSP_WIN_NO_EPOCH;
    }

    /* TODO: All the operations which we have not wrapped up will be failed, because they
     * are issued to user window. We need wrap up all operations.
     */

  fn_exit:
    return mpi_errno;

  fn_fail:
    goto fn_exit;
}

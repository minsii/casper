#include <stdio.h>
#include <stdlib.h>
#include "mtcore.h"

int MPI_Win_flush(int target_rank, MPI_Win win)
{
    MTCORE_Win *uh_win;
    int mpi_errno = MPI_SUCCESS;
    int user_rank, target_local_rank = -1;
    int j;

    MTCORE_DBG_PRINT_FCNAME();

    MTCORE_Fetch_uh_win_from_cache(win, uh_win);

    PMPI_Comm_rank(uh_win->user_comm, &user_rank);
#ifdef MTCORE_ENABLE_LOCAL_LOCK_OPT
    if (user_rank == target_rank && uh_win->is_self_locked) {
        int local_uh_rank;
        PMPI_Comm_rank(uh_win->local_uh_comm, &local_uh_rank);

        /* If target is itself, only flush the target on shared window.
         * It does not matter which window we are using for local communication,
         * we just choose local shared window in our implementation.
         */
        MTCORE_DBG_PRINT("[%d]flush self(%d, local_uh_win)\n", user_rank, local_uh_rank);
        mpi_errno = PMPI_Win_flush(local_uh_rank, uh_win->local_uh_win);
        if (mpi_errno != MPI_SUCCESS)
            goto fn_fail;
    }
    else
#endif
    {
        target_local_rank = uh_win->local_user_ranks[target_rank];

#ifdef MTCORE_ENABLE_SYNC_ALL_OPT

        /* Optimization for MPI implementations that have optimized lock_all.
         * However, user should be noted that, if MPI implementation issues lock messages
         * for every target even if it does not have any operation, this optimization
         * could lose performance and even lose asynchronous! */

        MTCORE_DBG_PRINT("[%d]flush_all(uh_wins[%d]), instead of target rank %d\n",
                         user_rank, target_local_rank, target_rank);
        mpi_errno = PMPI_Win_flush_all(uh_win->uh_wins[target_local_rank]);
        if (mpi_errno != MPI_SUCCESS)
            goto fn_fail;
#else
        for (j = 0; j < MTCORE_NUM_H; j++) {
            int target_h_rank_in_uh = uh_win->h_ranks_in_uh[target_rank * MTCORE_NUM_H + j];

            MTCORE_DBG_PRINT("[%d]flush(Helper(%d), uh_wins[%d]), instead of target rank %d\n",
                             user_rank, target_h_rank_in_uh, target_local_rank, target_rank);

            /* We flush target Helper processes in uh_window. Because for non-shared
             * targets, all translated operations are issued to target Helpers via uh_window.
             */
            mpi_errno = PMPI_Win_flush(target_h_rank_in_uh, uh_win->uh_wins[target_local_rank]);
            if (mpi_errno != MPI_SUCCESS)
                goto fn_fail;
        }
#endif
    }

    /* TODO: All the operations which we have not wrapped up will be failed, because they
     * are issued to user window. We need wrap up all operations.
     */

  fn_exit:
    return mpi_errno;

  fn_fail:
    goto fn_exit;
}

/**
* app.c
* BS Host Application Source File
*
*/
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include <dpu.h>
#include <dpu_log.h>
#include <unistd.h>
#include <getopt.h>
#include <assert.h>
#include <time.h>

#if ENERGY
#include <dpu_probe.h>
#endif

#include "params.h"
#include "timer.h"

// Define the DPU Binary path as DPU_BINARY here
#define DPU_BINARY "./bin/bs_dpu"

void reclamation_cb(struct dpu_set_t dpu_set, void *cb_args)
{
    unsigned int i = 0;
    struct dpu_set_t dpu;
    cb_arguments_t *cb_arguments = (cb_arguments_t *)cb_args;
    dpu_arguments_t *input_arguments = cb_arguments->input_arguments;

    uint64_t input_size = cb_arguments->input_size;
    uint64_t slice_per_dpu = cb_arguments->slice_per_dpu;
    DTYPE *input = cb_arguments->input;
    DTYPE *querys = cb_arguments->querys;
    static struct dpu_program_t *program = NULL;
    static uint32_t acc_nr_of_dpus = 0;
    uint32_t nr_of_dpus = 0;

    if (program)
        dpu_membo_load_with_program(dpu_set, DPU_BINARY, NULL, program, &program);
    else
        dpu_membo_load_with_program(dpu_set, DPU_BINARY, NULL, NULL, &program);

    // Copy input arrays
    DPU_FOREACH(dpu_set, dpu, i) {
        DPU_ASSERT(dpu_prepare_xfer(dpu, input_arguments));
    }
    DPU_ASSERT(dpu_push_xfer(dpu_set, DPU_XFER_TO_DPU, "DPU_INPUT_ARGUMENTS", 0, sizeof(*input_arguments), DPU_XFER_ASYNC));

    DPU_FOREACH(dpu_set, dpu, i)
    {
        DPU_ASSERT(dpu_prepare_xfer(dpu, input));
    }

    DPU_ASSERT(dpu_push_xfer(dpu_set, DPU_XFER_TO_DPU, DPU_MRAM_HEAP_POINTER_NAME, 0, input_size * sizeof(DTYPE), DPU_XFER_ASYNC));

    i = 0;

    DPU_FOREACH(dpu_set, dpu, i)
    {
        DPU_ASSERT(dpu_prepare_xfer(dpu, querys + slice_per_dpu * (i + acc_nr_of_dpus)));
    }

    DPU_ASSERT(dpu_push_xfer(dpu_set, DPU_XFER_TO_DPU, DPU_MRAM_HEAP_POINTER_NAME, input_size * sizeof(DTYPE), slice_per_dpu * sizeof(DTYPE), DPU_XFER_ASYNC));
    
    DPU_ASSERT(dpu_get_nr_dpus(dpu_set, &nr_of_dpus));
    acc_nr_of_dpus += nr_of_dpus;
}

// Create input arrays
void create_test_file(DTYPE * input, DTYPE * querys, uint64_t  nr_elements, uint64_t nr_querys) {

	input[0] = 1;
	for (uint64_t i = 1; i < nr_elements; i++) {
		input[i] = input[i - 1] + 1;
	}
	for (uint64_t i = 0; i < nr_querys; i++) {
		querys[i] = i;
	}
}

// Compute output in the host
int64_t binarySearch(DTYPE * input, DTYPE * querys, DTYPE input_size, uint64_t num_querys)
{
	uint64_t result = -1;
	DTYPE r;
	for(uint64_t q = 0; q < num_querys; q++)
	{
		DTYPE l = 0;
		r = input_size;
		while (l <= r) {
			DTYPE m = l + (r - l) / 2;

			// Check if x is present at mid
			if (input[m] == querys[q])
			result = m;

			// If x greater, ignore left half
			if (input[m] < querys[q])
			l = m + 1;

			// If x is smaller, ignore right half
			else
			r = m - 1;
		}
	}
	return result;
}


// Main of the Host Application
int main(int argc, char **argv) {

	struct Params p = input_params(argc, argv);
	struct dpu_set_t dpu_set, dpu;
	uint32_t nr_of_dpus;
	uint64_t input_size = INPUT_SIZE;
	uint64_t num_querys = p.num_querys;
	DTYPE result_host = -1;
	DTYPE result_dpu  = -1;
    FILE *fp;

	// Create the timer
	Timer timer;

	start(&timer, 5, 0);
	// Allocate DPUs and load binary
	//DPU_ASSERT(dpu_load(dpu_set, DPU_BINARY, NULL));
	//DPU_ASSERT(dpu_get_nr_dpus(dpu_set, &nr_of_dpus));

    nr_of_dpus = NR_DPUS;

	#if ENERGY
	struct dpu_probe_t probe;
	DPU_ASSERT(dpu_probe_init("energy_probe", &probe));
	#endif

	// Query number adjustement for proper partitioning
	if(num_querys % (nr_of_dpus * NR_TASKLETS))
	num_querys = num_querys + (nr_of_dpus * NR_TASKLETS - num_querys % (nr_of_dpus * NR_TASKLETS));

	assert(num_querys % (nr_of_dpus * NR_TASKLETS) == 0 && "Input dimension");    // Allocate input and querys vectors

	DTYPE * input  = malloc((input_size) * sizeof(DTYPE));
	DTYPE * querys = malloc((num_querys) * sizeof(DTYPE));

	// Create an input file with arbitrary data
	create_test_file(input, querys, input_size, num_querys);
    
	// Compute host solution
	start(&timer, 0, 0);
#if VERIFY_WITH_CPU
	result_host = binarySearch(input, querys, input_size - 1, num_querys);
#endif
	stop(&timer, 0);

	// Create kernel arguments
	uint64_t slice_per_dpu          = num_querys / nr_of_dpus;
	dpu_arguments_t input_arguments = {input_size, slice_per_dpu, 0};

    // AME cb args
    cb_arguments_t cb_args;
    cb_args.input_arguments = &input_arguments;
    cb_args.slice_per_dpu = slice_per_dpu;
    cb_args.input_size = input_size;
    cb_args.input = input;
    cb_args.querys = querys;

	start(&timer, 4, 0);
	//DPU_ASSERT(dpu_alloc_direct_reclaim(NR_DPUS, NULL, &dpu_set));
    DPU_ASSERT(dpu_alloc_ranks_membo_dmp(nr_of_dpus / NR_DPUS_PER_RANK, NULL, &dpu_set, &reclamation_cb, (void *)&cb_args));
    stop(&timer, 4);

    DPU_ASSERT(dpu_get_nr_dpus(dpu_set, &nr_of_dpus));
    printf("Allocated %d DPU(s)\n", nr_of_dpus);
    //printf("NR_TASKLETS\t%d\tBL\t%d\n", NR_TASKLETS, BL);

	for (unsigned int rep = 0; rep < p.n_warmup + p.n_reps; rep++) {
		// Perform input transfers
		uint64_t i = 0;

		if (rep >= p.n_warmup)
		start(&timer, 1, rep - p.n_warmup);

        if (rep != 0) {
            printf("rep is %u\n", rep);
            DPU_FOREACH(dpu_set, dpu, i)
            {
                DPU_ASSERT(dpu_prepare_xfer(dpu, &input_arguments));
            }

            DPU_ASSERT(dpu_push_xfer(dpu_set, DPU_XFER_TO_DPU, "DPU_INPUT_ARGUMENTS", 0, sizeof(input_arguments), DPU_XFER_DEFAULT));

            i = 0;

            DPU_FOREACH(dpu_set, dpu, i)
            {
                DPU_ASSERT(dpu_prepare_xfer(dpu, input));
            }

            DPU_ASSERT(dpu_push_xfer(dpu_set, DPU_XFER_TO_DPU, DPU_MRAM_HEAP_POINTER_NAME, 0, input_size * sizeof(DTYPE), DPU_XFER_DEFAULT));

            i = 0;

            DPU_FOREACH(dpu_set, dpu, i)
            {
                DPU_ASSERT(dpu_prepare_xfer(dpu, querys + slice_per_dpu * i));
            }

            DPU_ASSERT(dpu_push_xfer(dpu_set, DPU_XFER_TO_DPU, DPU_MRAM_HEAP_POINTER_NAME, input_size * sizeof(DTYPE), slice_per_dpu * sizeof(DTYPE), DPU_XFER_DEFAULT));
        }

		if (rep >= p.n_warmup)
		stop(&timer, 1);

		// Run kernel on DPUs
		if (rep >= p.n_warmup)
		{
			start(&timer, 2, rep - p.n_warmup);
			#if ENERGY
			DPU_ASSERT(dpu_probe_start(&probe));
			#endif
		}

		DPU_ASSERT(dpu_launch(dpu_set, DPU_SYNCHRONOUS));

		if (rep >= p.n_warmup)
		{
			stop(&timer, 2);
			#if ENERGY
			DPU_ASSERT(dpu_probe_stop(&probe));
			#endif
		}
		// Print logs if required
		#if PRINT
		unsigned int each_dpu = 0;
		printf("Display DPU Logs\n");
		DPU_FOREACH(dpu_set, dpu)
		{
			printf("DPU#%d:\n", each_dpu);
			DPU_ASSERT(dpulog_read_for_dpu(dpu.dpu, stdout));
			each_dpu++;
		}
		#endif

		// Retrieve results
		if (rep >= p.n_warmup)
		start(&timer, 3, rep - p.n_warmup);
		dpu_results_t* results_retrieve[nr_of_dpus];
		i = 0;
		DPU_FOREACH(dpu_set, dpu, i)
		{
			results_retrieve[i] = (dpu_results_t*)malloc(NR_TASKLETS * sizeof(dpu_results_t));
			DPU_ASSERT(dpu_prepare_xfer(dpu, results_retrieve[i]));
		}

		DPU_ASSERT(dpu_push_xfer(dpu_set, DPU_XFER_FROM_DPU, "DPU_RESULTS", 0, NR_TASKLETS * sizeof(dpu_results_t), DPU_XFER_DEFAULT));

		DPU_FOREACH(dpu_set, dpu, i)
		{
			for(unsigned int each_tasklet = 0; each_tasklet < NR_TASKLETS; each_tasklet++)
			{
				if(results_retrieve[i][each_tasklet].found > result_dpu)
				{
					result_dpu = results_retrieve[i][each_tasklet].found;
				}
			}
			free(results_retrieve[i]);
		}
		if(rep >= p.n_warmup)
		stop(&timer, 3);
	}
    stop(&timer, 5);

	// Print timing results
	printf("CPU Version Time (ms): ");
	print(&timer, 0, p.n_reps);
	printf("CPU-DPU Time (ms): ");
	print(&timer, 1, p.n_reps);
	printf("DPU Kernel Time (ms): ");
	print(&timer, 2, p.n_reps);
	printf("DPU-CPU Time (ms): ");
	print(&timer, 3, p.n_reps);
	printf("\n");

    double reclamation_time = get(&timer, 4, 1);
    double total_time = get(&timer, 5, 1);
    double other_time = total_time - reclamation_time - get(&timer, 1, p.n_reps);

    fp = fopen("../membo_output.txt", "a");
    fprintf(fp, "BS(%u): Reclamation time: %f (ms); Other exe. time: %f (ms); Total exe. time %f (ms)\n", nr_of_dpus, reclamation_time, other_time, total_time);
    fclose(fp);
	#if ENERGY
	double energy;
	DPU_ASSERT(dpu_probe_get(&probe, DPU_ENERGY, DPU_AVERAGE, &energy));
	printf("DPU Energy (J): %f\t", energy * num_iterations);
	#endif

    int status = 1;
#if VERIFY_WITH_CPU
	status = (result_dpu == result_host);
	if (status) {
		printf("[" ANSI_COLOR_GREEN "OK" ANSI_COLOR_RESET "] results are equal\n");
	} else {
		printf("[" ANSI_COLOR_RED "ERROR" ANSI_COLOR_RESET "] results differ!\n");
	}
#endif

	free(input);
	DPU_ASSERT(dpu_free(dpu_set));

	return status ? 0 : 1;
}

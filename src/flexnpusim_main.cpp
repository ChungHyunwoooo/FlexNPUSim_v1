#include <systemc.h>

#include "system/flexnpusim_system.h"

int sc_main(int argc, char* argv[]) {
    return flexnpu_sim::system::run_flexnpusim_system(argc, argv);
}

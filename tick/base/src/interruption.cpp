#include "interruption.h"

#include <iostream>
#include <csignal>

std::atomic<bool> Interruption::flag_interrupt(false);

const char *Interruption::what() const noexcept {
    return "Process was interrupted with signal SIGINT";
}

namespace {

void signal_handler(int signum) {
    Interruption::set();
}

}

class InterruptionInit {
 public :
    InterruptionInit() {
        std::signal(SIGINT, signal_handler);
    }
};

InterruptionInit init;

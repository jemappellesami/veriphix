OPENQASM 2.0;
include "qelib1.inc";
qreg q458[3];
rx(5*pi/4) q458[0];
cx q458[2],q458[1];
rx(pi/2) q458[1];
cx q458[0],q458[1];
rx(pi/4) q458[1];

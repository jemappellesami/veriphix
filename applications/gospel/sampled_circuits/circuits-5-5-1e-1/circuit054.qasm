OPENQASM 2.0;
include "qelib1.inc";
qreg q55[5];
cx q55[4],q55[3];
cx q55[2],q55[3];
cx q55[1],q55[2];
cx q55[1],q55[0];

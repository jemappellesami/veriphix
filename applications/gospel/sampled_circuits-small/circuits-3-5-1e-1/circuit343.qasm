OPENQASM 2.0;
include "qelib1.inc";
qreg q344[3];
rx(pi) q344[2];
cx q344[1],q344[2];
cx q344[1],q344[0];

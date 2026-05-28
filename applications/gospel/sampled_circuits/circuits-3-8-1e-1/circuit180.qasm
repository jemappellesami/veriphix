OPENQASM 2.0;
include "qelib1.inc";
qreg q181[3];
cx q181[1],q181[2];
cx q181[2],q181[1];
cx q181[1],q181[2];
cx q181[1],q181[0];
rx(pi/4) q181[1];

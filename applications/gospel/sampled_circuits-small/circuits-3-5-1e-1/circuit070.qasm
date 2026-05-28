OPENQASM 2.0;
include "qelib1.inc";
qreg q71[3];
rx(pi/2) q71[0];
cx q71[0],q71[1];
cx q71[1],q71[2];
rx(pi/4) q71[0];

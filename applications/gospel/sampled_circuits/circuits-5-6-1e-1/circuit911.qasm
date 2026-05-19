OPENQASM 2.0;
include "qelib1.inc";
qreg q912[5];
cx q912[3],q912[4];
cx q912[2],q912[3];
cx q912[1],q912[2];
cx q912[1],q912[0];
rx(pi/4) q912[1];

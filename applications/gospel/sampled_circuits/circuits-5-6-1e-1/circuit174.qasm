OPENQASM 2.0;
include "qelib1.inc";
qreg q175[5];
cx q175[3],q175[4];
cx q175[3],q175[2];
cx q175[2],q175[1];
cx q175[1],q175[0];
rx(pi/4) q175[1];

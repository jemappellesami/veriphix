OPENQASM 2.0;
include "qelib1.inc";
qreg q442[5];
cx q442[3],q442[4];
cx q442[2],q442[3];
cx q442[1],q442[2];
cx q442[0],q442[1];
rx(pi/4) q442[1];

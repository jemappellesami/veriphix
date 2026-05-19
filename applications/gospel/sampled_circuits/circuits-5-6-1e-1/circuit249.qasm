OPENQASM 2.0;
include "qelib1.inc";
qreg q250[5];
cx q250[4],q250[3];
cx q250[3],q250[2];
cx q250[2],q250[1];
cx q250[0],q250[1];
rx(pi/4) q250[1];

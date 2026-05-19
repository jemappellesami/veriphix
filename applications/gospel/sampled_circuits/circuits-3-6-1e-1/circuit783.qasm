OPENQASM 2.0;
include "qelib1.inc";
qreg q784[3];
rx(pi) q784[2];
cx q784[2],q784[1];
cx q784[0],q784[1];
rx(pi/4) q784[1];

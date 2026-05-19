OPENQASM 2.0;
include "qelib1.inc";
qreg q964[5];
cx q964[3],q964[4];
cx q964[3],q964[2];
cx q964[2],q964[1];
cx q964[0],q964[1];
rx(pi/4) q964[1];

OPENQASM 2.0;
include "qelib1.inc";
qreg q640[5];
cx q640[4],q640[3];
cx q640[3],q640[2];
cx q640[1],q640[2];
cx q640[0],q640[1];
rx(pi/4) q640[1];

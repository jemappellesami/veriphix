OPENQASM 2.0;
include "qelib1.inc";
qreg q905[5];
rx(pi/2) q905[4];
cx q905[4],q905[3];
cx q905[2],q905[3];
cx q905[2],q905[1];
cx q905[0],q905[1];

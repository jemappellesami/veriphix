OPENQASM 2.0;
include "qelib1.inc";
qreg q905[6];
cx q905[0],q905[1];
cx q905[2],q905[1];
rx(5*pi/4) q905[0];
cx q905[1],q905[0];
rx(pi/4) q905[1];

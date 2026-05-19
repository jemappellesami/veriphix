OPENQASM 2.0;
include "qelib1.inc";
qreg q364[4];
cx q364[3],q364[2];
rz(pi/4) q364[3];
cx q364[3],q364[2];
cx q364[2],q364[1];
cx q364[1],q364[0];

OPENQASM 2.0;
include "qelib1.inc";
qreg q513[4];
rz(7*pi/4) q513[3];
cx q513[3],q513[2];
cx q513[1],q513[2];
cx q513[0],q513[1];

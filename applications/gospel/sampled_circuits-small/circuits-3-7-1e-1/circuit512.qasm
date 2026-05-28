OPENQASM 2.0;
include "qelib1.inc";
qreg q513[3];
cx q513[1],q513[0];
cx q513[0],q513[1];
cx q513[1],q513[0];
cx q513[2],q513[1];
cx q513[0],q513[1];

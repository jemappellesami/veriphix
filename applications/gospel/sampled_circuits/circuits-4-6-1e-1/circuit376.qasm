OPENQASM 2.0;
include "qelib1.inc";
qreg q377[4];
rz(5*pi/4) q377[3];
cx q377[3],q377[2];
cx q377[1],q377[2];
cx q377[0],q377[1];
rx(pi/4) q377[1];

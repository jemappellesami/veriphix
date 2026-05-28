OPENQASM 2.0;
include "qelib1.inc";
qreg q377[3];
rz(pi/2) q377[2];
rx(7*pi/4) q377[2];
rz(7*pi/4) q377[2];
cx q377[2],q377[1];
cx q377[0],q377[1];

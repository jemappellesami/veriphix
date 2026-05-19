OPENQASM 2.0;
include "qelib1.inc";
qreg q568[4];
cx q568[2],q568[1];
rz(pi) q568[1];
cx q568[3],q568[2];
cx q568[1],q568[2];
cx q568[1],q568[0];

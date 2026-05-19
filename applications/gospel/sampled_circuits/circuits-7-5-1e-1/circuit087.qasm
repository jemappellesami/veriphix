OPENQASM 2.0;
include "qelib1.inc";
qreg q88[7];
cx q88[4],q88[5];
cx q88[3],q88[4];
cx q88[2],q88[3];
cx q88[2],q88[1];
cx q88[1],q88[0];

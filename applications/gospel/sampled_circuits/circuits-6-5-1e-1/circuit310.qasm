OPENQASM 2.0;
include "qelib1.inc";
qreg q311[6];
cx q311[4],q311[5];
cx q311[3],q311[4];
cx q311[2],q311[3];
cx q311[1],q311[2];
cx q311[0],q311[1];

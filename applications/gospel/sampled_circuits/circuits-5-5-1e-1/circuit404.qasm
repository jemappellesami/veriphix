OPENQASM 2.0;
include "qelib1.inc";
qreg q405[5];
cx q405[4],q405[3];
cx q405[3],q405[2];
cx q405[1],q405[2];
cx q405[0],q405[1];

OPENQASM 2.0;
include "qelib1.inc";
qreg q446[5];
cx q446[1],q446[0];
cx q446[3],q446[4];
cx q446[2],q446[3];
cx q446[2],q446[1];
cx q446[0],q446[1];

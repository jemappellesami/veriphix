OPENQASM 2.0;
include "qelib1.inc";
qreg q763[5];
cx q763[2],q763[3];
cx q763[3],q763[4];
cx q763[3],q763[2];
cx q763[2],q763[1];
cx q763[0],q763[1];

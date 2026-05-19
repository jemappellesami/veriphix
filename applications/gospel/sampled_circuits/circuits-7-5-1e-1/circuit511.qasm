OPENQASM 2.0;
include "qelib1.inc";
qreg q512[7];
cx q512[4],q512[5];
cx q512[4],q512[3];
cx q512[3],q512[2];
cx q512[1],q512[2];
cx q512[1],q512[0];

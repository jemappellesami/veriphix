OPENQASM 2.0;
include "qelib1.inc";
qreg q512[6];
cx q512[2],q512[3];
cx q512[3],q512[4];
cx q512[3],q512[2];
cx q512[2],q512[1];
cx q512[1],q512[0];

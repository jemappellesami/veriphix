OPENQASM 2.0;
include "qelib1.inc";
qreg q641[7];
cx q641[4],q641[5];
cx q641[3],q641[4];
cx q641[3],q641[2];
cx q641[1],q641[2];
cx q641[1],q641[0];
